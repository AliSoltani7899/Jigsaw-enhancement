import os
import time
import numpy as np
from datetime import timedelta
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.nn import InstanceNorm2d, LeakyReLU, Tanh, Sigmoid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import deeplay as dl
import matplotlib.pyplot as plt
from glob import glob
import os

# ------------------------
# Dataset Definition
# ------------------------
class ColorizationDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256)):
        self.image_paths = image_paths
        self.image_size = image_size
        self.rgb_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        self.gray_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        gray = self.gray_transform(img)
        rgb = self.rgb_transform(img)
        return gray, rgb

# ------------------------
# Helpers
# ------------------------

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def denormalize(x): return (x*0.5 + 0.5).clip(0,1)

# Enable reproducibility
torch.manual_seed(42)
cudnn.benchmark = True

# ------------------------
# Data and loaders
# ------------------------
# image_dir = 'data/animal'
# all_paths = [os.path.join(image_dir,f) for f in os.listdir(image_dir)
#              if f.lower().endswith(('.jpg','.jpeg','.png'))]
# train_paths, test_paths = train_test_split(all_paths, test_size=0.2, random_state=42)
train_paths = glob('data/animal/cat/*.png') + glob('data/animal/dog/*.png')

test_paths  = glob('data/test/*.png')
# print(f'Train: {len(train_paths)} | Test: {len(test_paths)}')
# train_paths = "data/animal"
# test_paths = "data/test"

# print(f'Train: {len(train_paths)} | Test: {len(test_paths)}')
# # # Lowered resolution for memory
img_size=(256,256)
batch_size=16
train_ds = ColorizationDataset(train_paths, image_size=img_size)
test_ds = ColorizationDataset(test_paths, image_size=img_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True,  num_workers=8)

device = get_device(); 
#print('Device:', device)

# ------------------------
# Model setup
# ------------------------
# Reduced U-Net channel sizes
gen = dl.UNet2d(in_channels=1, channels=[16,32,64,128], out_channels=3)
# Gradient checkpointing to save memory
try:
    gen.enable_gradient_checkpointing()
except AttributeError:
    pass
# Norm + activations
from torch.nn import LeakyReLU, Tanh
gen['decoder','blocks',:-1].all.normalized(InstanceNorm2d)
gen['decoder','blocks',:-1,'activation'].configure(LeakyReLU, negative_slope=0.2)
gen['decoder','blocks',-1,'activation'].configure(Tanh)
# Build
gen.build().to(device)

# Discriminator
disc = dl.ConvolutionalNeuralNetwork(in_channels=4, hidden_channels=[8,16,32], out_channels=1)
disc['blocks',...,'layer'].configure(kernel_size=4,stride=2,padding=1)
disc['blocks',...,'activation#-1'].configure(LeakyReLU, negative_slope=0.2)
disc['blocks',1:-1].all.normalized(InstanceNorm2d)
disc['blocks',-1,'activation'].configure(Sigmoid)
disc.build().to(device)

# ------------------------
# Losses & optimizers with mixed precision
# ------------------------
loss_disc = torch.nn.MSELoss()
loss_recon = torch.nn.L1Loss()
loss_percep = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
optim_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5,0.999))
optim_d = torch.optim.Adam(disc.parameters(), lr=5e-5, betas=(0.5,0.999))
scaler = torch.amp.GradScaler()

# ------------------------
# Training steps
# ------------------------

def train_disc(gm, dm, data):
    gray, rgb = data
    gray, rgb = gray.to(device), rgb.to(device)
    dm.train(); optim_d.zero_grad()
    with torch.cuda.amp.autocast():
        real_out = dm(torch.cat([gray,rgb],1))
        fake_rgb = gm(gray).detach()
        fake_out = dm(torch.cat([gray,fake_rgb],1))
        loss = 0.5*(loss_disc(real_out, torch.ones_like(real_out))
                   + loss_disc(fake_out, torch.zeros_like(fake_out)))
    scaler.scale(loss).backward(); scaler.step(optim_d); return loss.item()


def train_gen(gm, dm, data):
    gray, rgb = data
    gray, rgb = gray.to(device), rgb.to(device)
    gm.train(); optim_g.zero_grad()
    with torch.cuda.amp.autocast():
        fake_rgb = gm(gray)
        disc_out = dm(torch.cat([gray, fake_rgb], dim=1))
        adv = loss_disc(disc_out, torch.ones_like(disc_out))
        rec = loss_recon(fake_rgb, rgb)
        perc = loss_percep(fake_rgb, rgb)
        loss = adv + 100 * rec + 10 * perc
    scaler.scale(loss).backward()
    scaler.step(optim_g)
    scaler.update()
    return loss.item()

# ------------------------
# Evaluation
# ------------------------
def eval_and_show(batch):
    gray, rgb = batch
    gen.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        fake = gen(gray.to(device))
    gen.train()
    inp = denormalize(gray[0].permute(1,2,0).numpy())[...,0]
    pred = denormalize(fake[0].permute(1,2,0).cpu().numpy())
    tgt = denormalize(rgb[0].permute(1,2,0).numpy())
    fig, axs = plt.subplots(1,3,figsize=(9,3))
    axs[0].imshow(inp,'gray'); axs[1].imshow(pred); axs[2].imshow(tgt)
    for ax,t in zip(axs,['Input','Gen','GT']): ax.set_title(t); ax.axis('off')
    plt.show()

# ------------------------
# Main loop
# ------------------------

def eval_and_show(batch, save_path='output.png'):
    gray, rgb = batch
    gen.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        fake = gen(gray.to(device))
    gen.train()

    # Denormalize and convert to float32
    inp = denormalize(gray[0].permute(1,2,0).numpy()).astype(np.float32)[...,0]
    pred = denormalize(fake[0].permute(1,2,0).cpu().numpy()).astype(np.float32)
    tgt = denormalize(rgb[0].permute(1,2,0).numpy()).astype(np.float32)

    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(inp, cmap='gray')
    axs[0].set_title('Input (gray)')
    axs[0].axis('off')

    axs[1].imshow(pred)
    axs[1].set_title('Generated (rgb)')
    axs[1].axis('off')

    axs[2].imshow(tgt)
    axs[2].set_title('Ground Truth (rgb)')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

# ------------------------
# Main loop
# ------------------------
def main():
    epochs=200
    for ep in range(1,epochs+1):
        print(f'\nEpoch {ep}/{epochs}')
        start=time.time(); dls, gls = [],[]
        for batch in train_loader:
            dls.append(train_disc(gen,disc,batch))
            gls.append(train_gen(gen,disc,batch))
        print(f'Ep{ep}/{epochs} D{np.mean(dls):.4f} G{np.mean(gls):.4f} {timedelta(seconds=time.time()-start)}')
        eval_and_show(next(iter(test_loader)), save_path=f'outputs/output_{ep}.png')
        if ep%10==0: torch.save(gen.state_dict(),f'checkpoints/gen{ep}.pth')

if __name__ == '__main__':
    main()