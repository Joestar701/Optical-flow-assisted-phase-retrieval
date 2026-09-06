import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ====================== DATASET ======================
class InterferogramDataset(Dataset):
    """读取干涉训练集，返回 (pair_tensor [2,H,W], vibphase [1])"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = []
        for f in os.listdir(data_dir):
            if f.lower().endswith('_pic1.bmp'):
                base = f[:-9]  # 去掉 '_pic1.bmp'
                pic2 = self.data_dir / (base + '_pic2.bmp')
                params = self.data_dir / (base + '_params.txt')
                if pic2.exists() and params.exists():
                    self.samples.append(base)
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {data_dir}")
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def _read_vibphase(self, param_file):
        vib = 0.0
        with open(param_file, 'r') as f:
            for line in f:
                if line.startswith('vibphase'):
                    vib = float(line.strip().split('=')[1])
                    break
        return vib

    def __getitem__(self, idx):
        base = self.samples[idx]
        img1_path = self.data_dir / (base + '_pic1.bmp')
        img2_path = self.data_dir / (base + '_pic2.bmp')
        param_file = self.data_dir / (base + '_params.txt')

        img1 = np.array(Image.open(img1_path).convert('L'), dtype=np.float32) / 255.0
        img2 = np.array(Image.open(img2_path).convert('L'), dtype=np.float32) / 255.0

        # per-image normalization
        img1 = (img1 - img1.mean()) / (img1.std() + 1e-6)
        img2 = (img2 - img2.mean()) / (img2.std() + 1e-6)

        pair = np.stack([img1, img2], axis=0).astype(np.float32)
        pair_tensor = torch.from_numpy(pair)
        vib_tensor = torch.tensor([self._read_vibphase(param_file)], dtype=torch.float32)
        return pair_tensor, vib_tensor

# ====================== RAFT VIBPHASE NETWORK ======================
class BasicEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, output_dim, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class ContextNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

class AllPairsCorrelation(nn.Module):
    def forward(self, fmap1, fmap2):
        B, C, H, W = fmap1.shape
        N = H*W
        f1 = fmap1.view(B, C, N)
        f2 = fmap2.view(B, C, N)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        corr = torch.bmm(f1.permute(0,2,1), f2)
        corr = corr.view(B, N, H, W)
        return corr

class CorrSampler(nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, corr, H, W, device):
        B, N, _, _ = corr.shape
        r = self.radius
        yy = torch.arange(0, H, device=device)
        xx = torch.arange(0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        base_idx = (grid_y * W + grid_x).view(-1)
        neighbor_idxs = []
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                ny = torch.clamp(grid_y + dy, 0, H-1)
                nx = torch.clamp(grid_x + dx, 0, W-1)
                idx = (ny * W + nx).view(-1)
                neighbor_idxs.append(idx)
        neighbor_idxs = torch.stack(neighbor_idxs, dim=0).long()
        K = neighbor_idxs.shape[0]
        B_out = []
        corr_flat = corr.view(B, N, -1)
        for b in range(B):
            corr_flat_b = corr_flat[b]
            cv = []
            for k in range(K):
                idxs = neighbor_idxs[k]
                vals = corr_flat_b[torch.arange(N, device=device), idxs]
                cv.append(vals.view(H, W))
            B_out.append(torch.stack(cv, dim=0))
        return torch.stack(B_out, dim=0)

class ConvGRUUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.convz = nn.Conv2d(input_dim+hidden_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(input_dim+hidden_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(input_dim+hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            return torch.tanh(self.convq(x))
        inp = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.convz(inp))
        r = torch.sigmoid(self.convr(inp))
        q = torch.tanh(self.convq(torch.cat([x, r*h], dim=1)))
        new_h = (1-z)*h + z*q
        return new_h

class FlowHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim//2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_dim//2, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)

def upsample_flow(flow, scale_factor=8.0):
    return F.interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=True)

class RAFTVibPhase(nn.Module):
    def __init__(self, iters=12, corr_radius=4, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.iters = iters
        self.encoder = BasicEncoder(1, feature_dim)
        self.context = ContextNetwork(1, hidden_dim)
        self.allpairs = AllPairsCorrelation()
        self.corr_sampler = CorrSampler(corr_radius)
        K = (2*corr_radius+1)**2
        self.corr_proc = nn.Sequential(
            nn.Conv2d(K, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.update = ConvGRUUpdate(hidden_dim, hidden_dim)
        self.flow_head = FlowHead(hidden_dim)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, pair):
        im1 = pair[:,0:1,:,:]
        im2 = pair[:,1:2,:,:]
        B, _, H, W = pair.shape
        device = pair.device

        fmap1 = self.encoder(im1)
        fmap2 = self.encoder(im2)
        _, Cf, h, w = fmap1.shape

        corr_all = self.allpairs(fmap1, fmap2)
        cost_vol = self.corr_sampler(corr_all, h, w, device)
        cost_feat = self.corr_proc(cost_vol)

        ctx = self.context(im1)
        if ctx.shape[2:] != cost_feat.shape[2:]:
            ctx = F.interpolate(ctx, size=cost_feat.shape[2:], mode='bilinear', align_corners=True)

        hidden = ctx
        flow = torch.zeros(B, 2, h, w, device=device)

        for _ in range(self.iters):
            hidden = self.update(cost_feat, hidden)
            delta_flow = self.flow_head(hidden)
            flow = flow + delta_flow

        flow_up = upsample_flow(flow)
        vib = self.regressor(hidden).squeeze(1)
        return flow_up, vib

# --------------------------- LOSS ---------------------------
class VibLoss(nn.Module):
    """L1 loss for vibphase regression, optionally with flow loss."""
    def __init__(self, flow_weight=0.0):
        super().__init__()
        self.flow_weight = flow_weight
        self.l1 = nn.L1Loss()

    def forward(self, vib_pred, vib_gt, flow_pred=None, flow_gt=None):
        # reshape to [B] to avoid broadcasting issues
        vib_pred = vib_pred.view(-1)
        vib_gt = vib_gt.view(-1)
        loss_v = self.l1(vib_pred, vib_gt)
        loss = loss_v

        # optional flow loss
        if self.flow_weight > 0 and (flow_pred is not None) and (flow_gt is not None):
            # flatten flow to [B, 2*H*W] or keep original shape
            loss_f = self.l1(flow_pred, flow_gt)
            loss = loss + self.flow_weight * loss_f

        return loss, loss_v


# ====================== TRAIN / VALIDATE ======================
def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total = 0.0
    pbar = tqdm(loader)
    for pair, vib in pbar:
        pair, vib = pair.to(device), vib.to(device)
        optimizer.zero_grad()
        flow_pred, vib_pred = model(pair)
        loss, l_v = loss_fn(vib_pred, vib)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        pbar.set_description(f"train_loss={total/(pbar.n+1):.6f}")
    return total / len(loader)

def validate_epoch(model, loader, device, loss_fn):
    model.eval()
    total, total_v = 0.0, 0.0
    with torch.no_grad():
        for pair, vib in loader:
            pair, vib = pair.to(device), vib.to(device)
            flow_pred, vib_pred = model(pair)
            loss, l_v = loss_fn(vib_pred, vib)
            total += loss.item()
            total_v += l_v.item()
    return total / len(loader), total_v / len(loader)

# ====================== MAIN ======================
def main():
    # -------- CONFIG --------
    data_dir = r"E:\Project\VSI\TrainData"
    out_dir = r"checkpoints"
    epochs = 30
    batch_size = 4
    lr = 1e-4
    iters = 12
    radius = 4
    flow_weight = 0.0
    seed = 42

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # -------- DATASET & DATALOADER --------
    dataset = InterferogramDataset(data_dir)
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    split = max(1, int(0.1*n))
    train_ds = torch.utils.data.Subset(dataset, idx[split:])
    val_ds = torch.utils.data.Subset(dataset, idx[:split])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # -------- MODEL --------
    model = RAFTVibPhase(iters=iters, corr_radius=radius)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = VibLoss(flow_weight=flow_weight)
    os.makedirs(out_dir, exist_ok=True)

    best_val = 1e9
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, val_vib = validate_epoch(model, val_loader, device, loss_fn)
        print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}, val_vib={val_vib:.6f}")
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(ckpt, os.path.join(out_dir, f'raft_epoch_{epoch:03d}.pth'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(out_dir, 'best_raft_vib.pth'))

    print("Training complete!")

if __name__ == "__main__":
    main()
