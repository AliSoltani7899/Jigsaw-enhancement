import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import deeplay as dl
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class GrayDataset(Dataset):
    """
    A dataset that loads images from a directory (and subdirectories) as single-channel grayscale tensors.

    Args:
        root_dir (str or Path): Path to the directory containing images.
        extensions (tuple of str): Allowed file extensions, e.g. ('.jpg', '.png').
    """
    def __init__(self, root_dir, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        self.root_dir = Path(root_dir)
        # Recursively collect image file paths
        self.paths = sorted(
            [p for p in self.root_dir.rglob('*') if p.suffix.lower() in extensions]
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root_dir} with extensions {extensions}")

        # Grayscale + ToTensor transform
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # outputs [1, H, W] in [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')  # ensure 3-channel input for Grayscale
            clean = self.transform(img)
        return clean


class NoisyGrayDataset(Dataset):
    """
    A dataset that wraps GrayDataset and returns noisy and clean grayscale tensors.

    Args:
        root_dir (str or Path): Path to the directory containing images.
        noise_std (float): Standard deviation of Gaussian noise added to clean image.
    """
    def __init__(self, root_dir, noise_std=0.3, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        self.clean_ds = GrayDataset(root_dir, extensions)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.clean_ds)

    def __getitem__(self, idx):
        clean = self.clean_ds[idx]  # [1, H, W]
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean

# parameters
root_data_dir = "animal"  # <-- point this to your folder
batch_size    = 16
noise_std     = 0.3
lr            = 1e-3
num_epochs    = 20
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

dataset = NoisyGrayDataset(root_data_dir, noise_std=noise_std)

fraction = 1  # use 10% of the dataset; set to 0.2 for 20%
num_samples = int(len(dataset) * fraction)
indices = np.random.choice(len(dataset), num_samples, replace=False)
subset = Subset(dataset, indices)


loader  = dl.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4)

print(len(subset))

unet = dl.UNet2d(
    in_channels=1, channels=[16, 32, 64,64, 128], out_channels=1, skip=dl.Cat(),
)

    
def test(model):
    root_data_dir = "test"
    batch_size    = 32
    noise_std     = 0.3
    # --------------------

    # 1. loader
    test_set    = NoisyGrayDataset(root_data_dir, noise_std=noise_std)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # 2. fetch one batch and take first 5
    noisy_imgs, clean_imgs = next(iter(test_loader))  # [B,1,H,W]
    noisy_5, clean_5       = noisy_imgs[:5], clean_imgs[:5]

    # 3. plot 5 rows × 2 cols
    fig, axes = plt.subplots(5, 2, figsize=(6, 15))
    for i in range(5):
        # clean
        axc = axes[i, 0]
        axc.imshow(clean_5[i].squeeze().cpu().numpy(), cmap='gray')
        axc.set_title(f'Clean #{i+1}')
        axc.axis('off')
        # noisy
        axn = axes[i, 1]
        axn.imshow(noisy_5[i].squeeze().cpu().numpy(), cmap='gray')
        axn.set_title(f'Noisy #{i+1}')
        axn.axis('off')

    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))

    for i in range(5):
        input_img  = noisy_5[i]            # [1,H,W]
        target_img = clean_5[i]            # [1,H,W]
        pred_img   = model(input_img.unsqueeze(0)).detach()  # [1,1,H,W]

        # Convert to numpy for plotting
        inp_np  = input_img[0].cpu().numpy()
        tgt_np  = target_img[0].cpu().numpy()
        pred_np = pred_img[0, 0].cpu().numpy()

        # Plot input
        axes[i, 0].imshow(inp_np, cmap='gray')
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 0].axis('off')

        # Plot target
        axes[i, 1].imshow(tgt_np, cmap='gray')
        axes[i, 1].set_title(f"Target {i+1}")
        axes[i, 1].axis('off')

        # Plot prediction
        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title(f"Predicted {i+1}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    regressor_template = dl.Regressor(
        model=unet, loss=torch.nn.MSELoss(), optimizer=dl.Adam(lr=1e-3),
        )
    ed = regressor_template.create()
    ed_trainer = dl.Trainer(max_epochs=150, accelerator="cuda", devices=1)
    print("start")
    ed_trainer.fit(ed, loader)
    test(ed)
    # Save the model
    torch.save(ed.state_dict(), "denoiser_model.pth")
    print("Model saved as denoiser_model.pth")

if __name__ == "__main__":
    main()

