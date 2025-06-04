# Main
import os
import random
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from scipy.optimize import linear_sum_assignment
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# Define transformation for tile extraction
def get_tile_transform(tile_size, noise_std, augment=False):
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # [1, H, W], range [0,1]
    ]
    return transforms.Compose(transform_list)

# Tile extraction as given
def extract_tiles(img, grid_size, tile_size, overlap, augment, noise_std):
    H, W = grid_size
    w, h = img.size
    cell_w, cell_h = w / W, h / H
    transform = get_tile_transform(tile_size, noise_std, augment)
    tiles = []
    for i in range(H):
        for j in range(W):
            x0 = int(max(0, j*cell_w - overlap))
            y0 = int(max(0, i*cell_h - overlap))
            x1 = int(min(w, (j+1)*cell_w + overlap))
            y1 = int(min(h, (i+1)*cell_h + overlap))
            crop = img.crop((x0, y0, x1, y1)).resize((tile_size, tile_size), Image.BICUBIC)
            tiles.append(transform(crop))
    return torch.stack(tiles)  # shape: [H*W, 1, tile_size, tile_size]

# Dataset that returns tiles with noise
class TileNoisyGrayDataset(Dataset):
    def __init__(self, root_dir, grid_size=(4, 4), tile_size=64, overlap=0, noise_std=0.3, augment=False):
        self.base_dataset = GrayDataset(root_dir)
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.noise_std = noise_std
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        clean_img = self.base_dataset[idx]  # [1, H, W]
        img_pil = transforms.ToPILImage()(clean_img)

        clean_tiles = extract_tiles(img_pil, self.grid_size, self.tile_size, self.overlap, self.augment, noise_std=0)
        noise = torch.randn_like(clean_tiles) * self.noise_std
        noisy_tiles = torch.clamp(clean_tiles + noise, 0.0, 1.0)

        # Shuffle the tiles and get permutation
        num_tiles = noisy_tiles.size(0)
        perm = torch.randperm(num_tiles)
        shuffled_noisy_tiles = noisy_tiles[perm]

        return shuffled_noisy_tiles, perm

    
class LSCE(nn.Module):
    def __init__(self, eps=0.05):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        """
        logits: [B*N, N]
        target: [B*N]
        """
        logp = F.log_softmax(logits, dim=1)
        nll = -logp[torch.arange(target.size(0)), target]
        smooth = -logp.mean(dim=1)
        return ((1 - self.eps) * nll + self.eps * smooth).mean()
    
class JigsawModel(nn.Module):
    def __init__(
        self,
        grid_size=(3, 3),
        backbone='resnet50',
        weights=ResNet50_Weights.IMAGENET1K_V2,
        nhead=8,
        num_layers=2,
        dropout=0.1,
        noise_std=0.05,
    ):
        super().__init__()
        H, W = grid_size
        self.N = H * W

        # --- Gaussian noise augmentation for noisy tiles ---
        #self.noise = GaussianNoise(std=noise_std)

        # --- CNN encoder (ResNet50 trunk w/o final FC) adapted for 1-channel input ---
        base = getattr(models, backbone)(weights=weights)
        # Replace first conv to accept single-channel input
        orig_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=(orig_conv.bias is not None)
        )
        # Initialize new conv weights by averaging across original RGB channels
        with torch.no_grad():
            new_conv.weight[:] = orig_conv.weight.mean(dim=1, keepdim=True)
            if orig_conv.bias is not None:
                new_conv.bias[:] = orig_conv.bias
        base.conv1 = new_conv

        # Remove final FC
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.embed_dim = base.fc.in_features

        # --- Learnable positional embeddings, one per slot ---
        self.pos_emb = nn.Parameter(torch.zeros(1, self.N, self.embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # --- Transformer Encoder over the sequence of tile embeddings ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # --- Per-token MLP head to predict slot index 0…N-1 ---
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.N)
        )

    def forward(self, x):
        """
        x: Tensor[B, N, 1, T, T] (grayscale tiles)
        returns logits: Tensor[B, N, N]
        """
        B, N, C, T, _ = x.shape
        # 1) Add noise to each tile
        x = x.view(B * N, C, T, T).view(B, N, C, T, T)

        # 2) CNN encode each tile → (B*N, D)
        feats = self.encoder(x.view(B * N, C, T, T)).view(B, N, -1)

        # 3) Add positional embedding
        feats = feats + self.pos_emb

        # 4) Contextualize with Transformer
        feats = self.transformer(feats)

        # 5) Predict a distribution over target slot for each input tile
        logits = self.head(feats)
        return logits

    @torch.no_grad()
    def hungarian_assign(self, logits, temperature=1.0):
        """
        Solve assignment per batch element via Hungarian on -logit scores.
        Returns: Tensor[B, N] giving the assigned slot index for each tile.
        """
        scores = (logits / temperature).cpu().numpy()
        perms = []
        for mat in scores:
            _, col_ind = linear_sum_assignment(-mat)
            perms.append(torch.tensor(col_ind, dtype=torch.long))
        return torch.stack(perms, dim=0)


def train_one_epoch(model, loader, optim, scheduler, epoch, freeze_epochs=5):
    model.train()
    # freeze encoder if desired
    for p in model.encoder.parameters():
        p.requires_grad = (epoch > freeze_epochs)

    running_loss = 0.0
    running_correct = 0
    running_tiles = 0

    for tiles, perm in tqdm(loader, desc=f"Train Epoch {epoch}"):
        tiles, perm = tiles.to(device), perm.to(device)

        optim.zero_grad()
        logits = model(tiles)                # [B, N, N]
        B, N, _ = logits.shape

        # --- loss ---
        loss = LSCE()(logits.view(B*N, -1), perm.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        # --- accuracy ---
        with torch.no_grad():
            preds = model.hungarian_assign(logits)  # [B, N], on CPU
            preds = preds.to(device)
            running_correct += (preds == perm).sum().item()
            running_tiles   += B * N

        running_loss += loss.item() * B

    scheduler.step()

    avg_loss = running_loss / len(loader.dataset)
    accuracy = running_correct / running_tiles
    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch_imagewise(model, loader, device):
    """
    Returns:
      avg_loss   : float  — average LSCE loss per image
      tile_acc   : float  — total_correct_tiles / total_tiles
      image_acc  : float  — images_with_all_tiles_correct / total_images
    """
    model.eval()
    criterion = LSCE()
    total_loss = 0.0
    total_tiles = 0
    correct_tiles = 0
    total_images = 0
    correct_images = 0

    for tiles, perm in loader:
        tiles, perm = tiles.to(device), perm.to(device)  # [B, N, C, T, T], [B, N]
        B, N = perm.shape

        logits = model(tiles)                           # [B, N, N]
        # accumulate loss *per‑image*
        total_loss += criterion(logits.view(B*N, -1),
                                 perm.view(-1)).item() * B

        # compute assignments
        preds = model.hungarian_assign(logits).to(device)  # [B, N]
        eq = (preds == perm)                               # [B, N]

        # tile‑level accuracy
        correct_tiles += eq.sum().item()
        total_tiles   += B * N

        # image‑level (all‑correct) accuracy
        correct_images += eq.all(dim=1).sum().item()
        total_images   += B

    avg_loss  = total_loss  / len(loader.dataset)
    tile_acc  = correct_tiles / total_tiles
    image_acc = correct_images / total_images

    return avg_loss, tile_acc, image_acc




def main():

    epochs, bs = 300, 8
    grid_size  = (3,3)
    tile_size  = 96
    overlap    = 0       # fixed
    noise_std  = 0.1

    # Model, optimizer, scheduler
    model = JigsawModel(grid_size=grid_size, num_layers=2, nhead=8, dropout=0.1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs-5)
    sched  = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warmup, cosine], milestones=[5])

    # Datasets & loaders
    train_ds = TileNoisyGrayDataset('data/animal', grid_size, tile_size, overlap,  noise_std)
    val_ds   = TileNoisyGrayDataset('data/test',   grid_size, tile_size, overlap, noise_std)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=9, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=9, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    best_val_acc = 0.0
    early_stop_patience = 5
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_dl, optim, sched, epoch)
        val_loss,   val_acc   , image_acc= validate_one_epoch_imagewise(model, val_dl, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4%} | "
            f"image_acc for val: {image_acc:.4f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            print(f"→ New best model saved (acc={val_acc:.4%})")
            print(f"best acc: {best_val_acc}")
            torch.save(model.state_dict(), f'checks/model_2L_{epoch}_{train_acc:.3%}_{val_acc:.3%}.pth')
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Stopping early at epoch {epoch}, no improvement for {early_stop_patience} epochs.")
                break

        with open('training_log_CD_2L.txt', 'a') as f:
            f.write(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4%}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4%}, "
                f"image_acc for val: {image_acc:.4f}\n")

if __name__ == '__main__':
    main()
