"""
Full training script for flood segmentation using:
- CBAM (channel then spatial attention)
- ASPP (multi-scale dilation + image pooling)
- SelfAttention (non-local block)
- ImprovedCNNDecoder (U-Net style upsampling)
- ImprovedDualBranchDecoder (CNN + Transformer with attention-guided fusion)

This file is self-contained: model + loss + metrics + train/val loops + CLI.

Requirements:
    pip install timm segmentation-models-pytorch torch torchvision

Dataset expectation:
    Implement/adjust the FloodDataset to return (sar, optical, mask) where
      sar:     FloatTensor [1,H,W] in range ~[-1,1] or [0,1]
      optical: FloatTensor [1,H,W] (if you have RGB, adapt to 3 and modify code)
      mask:    FloatTensor [1,H,W] with 0/1 values

If you already have a FloodDataset in dataset.py you can import that instead
of the placeholder below by setting --use_external_dataset.
"""

from __future__ import annotations
import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
import timm


# =====================
#  Attention Modules
# =====================
class ChannelAttention(nn.Module):
    """CBAM Channel Attention using both avg and max pooling with shared MLP."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx, _ = torch.max(x, dim=2, keepdim=True)
        mx, _ = torch.max(mx, dim=3, keepdim=True)
        att = self.mlp(avg) + self.mlp(mx)
        return self.sigmoid(att)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention with 7x7 conv on [max;avg] maps."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([mx, avg], dim=1)
        return self.sigmoid(self.conv(a))


class CBAM(nn.Module):
    """Sequential Channel then Spatial attention refinement."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =====================
#  ASPP Module
# =====================
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rates[1],
                        dilation=rates[1],
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rates[2],
                        dilation=rates[2],
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rates[3],
                        dilation=rates[3],
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        gp = self.image_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gp)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class ASPP_CBAM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.aspp = ASPP(in_channels, out_channels)
        self.cbam = CBAM(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.aspp(x))


# =====================
#  Self-Attention (Non-local)
# =====================
class SelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        inter = max(in_channels // 8, 8)
        self.query = nn.Conv2d(in_channels, inter, 1)
        self.key = nn.Conv2d(in_channels, inter, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)  # [B, HW, C']
        k = self.key(x).view(b, -1, h * w)  # [B, C', HW]
        attn = self.softmax(torch.bmm(q, k))  # [B, HW, HW]
        v = self.value(x).view(b, c, h * w)  # [B, C, HW]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


# =====================
#  Decoders
# =====================
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ImprovedCNNDecoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.up1 = UpBlock(in_channels, 256)  # 1/16 -> 1/8
        self.up2 = UpBlock(256, 128)  # 1/8  -> 1/4
        self.up3 = UpBlock(128, 64)  # 1/4  -> 1/2
        self.up4 = UpBlock(64, 32)  # 1/2  -> 1/1
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out(x)


class TransformerDecoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.up1 = UpBlock(in_channels, 256)
        self.up2 = UpBlock(256, 256)
        self.attn = SelfAttention(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        return self.attn(x)  # [B,256,H/4,W/4]


class ImprovedDualBranchDecoder(nn.Module):
    """Fuse high-res CNN mask with transformer features using attention-guided fusion."""

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.cnn = ImprovedCNNDecoder(in_channels)
        self.tf = TransformerDecoder(in_channels)
        # Upsample transformer to full res and reduce channels
        self.tf_up = nn.Sequential(
            UpBlock(256, 128),  # x2
            UpBlock(128, 64),  # x2
            UpBlock(64, 32),  # x2
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # Attention-guided fusion
        # Predict a soft gate alpha in [0,1] from stacked logits [m_cnn; m_tf]
        # Note: final model should output logits (no sigmoid) because we use BCEWithLogits.
        self.fuse_attn = nn.Conv2d(2, 1, kernel_size=1, bias=True)  # gate logits
        # Light refinement head over fused logits
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),  # output logits
        )

    def forward(self, x):
        # Run both branches on the same input
        m_cnn = self.cnn(x)  # CNN branch output (logits)
        tf_features = self.tf(x)  # Transformer branch features [B,256,H/4,W/4]
        m_tf = self.tf_up(tf_features)  # Upsample transformer to full res (logits)

        # Make sure both branches have same spatial size
        if m_cnn.shape[-2:] != m_tf.shape[-2:]:
            m_tf = F.interpolate(
                m_tf, size=m_cnn.shape[-2:], mode="bilinear", align_corners=False
            )

        # Predict a soft gate alpha and fuse branch logits
        stacked = torch.cat([m_cnn, m_tf], dim=1)  # [B,2,H,W]
        alpha = torch.sigmoid(self.fuse_attn(stacked))  # [B,1,H,W] in [0,1]
        fused_logits = alpha * m_cnn + (1.0 - alpha) * m_tf  # logits
        out = self.fuse_conv(fused_logits)  # refined logits
        return out


# =====================
#  Encoder & Full Model
# =====================
class AdvancedResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True
        )
        feat_ch = self.backbone.feature_info.channels()[-1]
        # Map SAR/Optical 1ch -> 3ch for ResNet
        self.sar_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.opt_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.sar_adapt.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.opt_adapt.weight, a=math.sqrt(5))
        # Fuse and reduce to 512
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_ch * 2, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(512)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        s = self.backbone(self.sar_adapt(sar))[-1]
        o = self.backbone(self.opt_adapt(optical))[-1]
        x = torch.cat([s, o], dim=1)
        x = self.reduce(x)
        x = self.cbam(x)
        return x  # [B,512,H/16,W/16] for resnet50


class AdvancedFloodSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        use_aspp: bool = True,
        use_dual_decoder: bool = True,
    ):
        super().__init__()
        self.encoder = AdvancedResNetFeatureExtractor(encoder_name, pretrained=True)
        self.use_aspp = use_aspp
        if use_aspp:
            self.ms = ASPP_CBAM(512, 512)
        else:
            self.ms = nn.Identity()
        self.decoder = (
            ImprovedDualBranchDecoder(512)
            if use_dual_decoder
            else ImprovedCNNDecoder(512)
        )
        self.dropout = nn.Dropout2d(0.1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        x = self.encoder(sar, optical)
        x = self.ms(x)
        x = self.decoder(x)
        return x


# =====================
#  Loss & Metrics
# =====================
class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0, smooth: float = 1e-6):
        super().__init__()
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.pos_weight.to(pred.device)
        )
        # Per-sample Dice, skip empties
        pred_prob = torch.sigmoid(pred)
        B = pred.shape[0]
        pred_flat = pred_prob.view(B, -1)
        target_flat = target.view(B, -1)
        inter = (pred_flat * target_flat).sum(dim=1)
        sums = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        valid = (target_flat.sum(dim=1) > 0).float()
        dice = (2 * inter + self.smooth) / (sums + self.smooth)
        dice = (dice * valid).sum() / (valid.sum() + 1e-6)
        dice_loss = 1.0 - dice
        return bce + dice_loss


def batch_metrics(
    pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5
) -> Dict[str, float]:
    pred_prob = torch.sigmoid(pred)
    pred_bin = (pred_prob > thr).float()
    B = pred.shape[0]
    f1s = []
    for i in range(B):
        t = target[i]
        p = pred_bin[i]
        if t.sum() == 0:
            f1s.append(1.0 if p.sum() == 0 else 0.0)
            continue
        tp = ((p == 1) & (t == 1)).sum().item()
        fp = ((p == 1) & (t == 0)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return {"f1": float(sum(f1s) / max(len(f1s), 1))}


# =====================
#  (Optional) Minimal Dataset Placeholder
# =====================
class FloodDatasetPlaceholder(Dataset):
    """Replace this with your real dataset by using --use_external_dataset and importing your class."""

    def __init__(self, n=32, H=512, W=512):
        super().__init__()
        self.n, self.H, self.W = n, H, W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sar = torch.randn(1, self.H, self.W)
        opt = torch.randn(1, self.H, self.W)
        mask = (torch.rand(1, self.H, self.W) > 0.95).float()
        return sar, opt, mask


# =====================
#  Train / Validate
# =====================
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    out_dir: str,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    tot_loss, n = 0.0, 0
    all_preds, all_targets = [], []

    for sar, opt, m in loader:
        sar, opt, m = (
            sar.to(device, non_blocking=True),
            opt.to(device, non_blocking=True),
            m.to(device, non_blocking=True),
        )
        pred = model(sar, opt)
        if pred.shape[-2:] != m.shape[-2:]:
            pred = F.interpolate(
                pred, size=m.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = criterion(pred, m)
        tot_loss += float(loss.item())
        n += 1
        
        # Store predictions and targets for overall metric calculation
        all_preds.append(torch.sigmoid(pred).cpu())
        all_targets.append(m.cpu())

    # Calculate metrics over the entire validation set
    val_f1 = 0.0
    if all_preds:
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        # Use a single call to batch_metrics if it supports tensors of all batches
        # Or implement the logic here
        pred_bin = (all_preds_tensor > 0.5).float()
        target_flat = all_targets_tensor.view(-1)
        pred_flat = pred_bin.view(-1)
        
        if target_flat.sum() > 0:
            tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
            fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
            fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            val_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        else: # No positive labels in the entire validation set
            val_f1 = 1.0 if pred_flat.sum() == 0 else 0.0


    # Save a small grid of predictions for quick inspection
    os.makedirs(out_dir, exist_ok=True)
    try:
        sar0, opt0, m0 = next(iter(loader))
        sar0, opt0 = sar0.to(device), opt0.to(device)
        with torch.no_grad():
            pr0 = torch.sigmoid(model(sar0, opt0))
            grid = torch.cat(
                [sar0[:4].cpu(), opt0[:4].cpu(), m0[:4].cpu(), pr0[:4].cpu()], dim=0
            )
            save_image(grid, os.path.join(out_dir, f"val_epoch{epoch:03d}.png"), nrow=4)
    except Exception:
        pass
    return tot_loss / max(n, 1), val_f1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    if args.use_external_dataset:
        from dataset import FloodDataset  # your implementation

        full_ds = FloodDataset(args.sar_dir, args.opt_dir, args.mask_dir)
    else:
        full_ds = FloodDatasetPlaceholder(n=64)

    val_size = max(10, int(args.val_split * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = AdvancedFloodSegmentationModel(
        encoder_name=args.encoder, use_aspp=True, use_dual_decoder=True
    ).to(device)

    # Loss & Optimizer
    criterion = BCEDiceLoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Train loop
    best_f1 = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for sar, opt, m in loop:
            sar, opt, m = (
                sar.to(device, non_blocking=True),
                opt.to(device, non_blocking=True),
                m.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(sar, opt)
                if pred.shape[-2:] != m.shape[-2:]:
                    pred = F.interpolate(
                        pred, size=m.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss = criterion(pred, m)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            loop.set_postfix(train_loss=running / max(1, loop.n))

        val_loss, val_f1 = validate(
            model, val_loader, criterion, device, args.out_dir, epoch
        )
        scheduler.step(val_f1)
        print(
            f"[Epoch {epoch}] Train Loss: {running/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
            print(f"✅ Saved new best (F1={best_f1:.4f})")

    print("Training complete. Best F1:", best_f1)


# =====================
#  CLI
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_external_dataset",
        action="store_true",
        help="Use your dataset.FloodDataset instead of placeholder",
    )
    parser.add_argument(
        "--sar_dir",
        type=str,
        default="D:/Flood-Segmentation/dataset/HandLabeled/S1Hand",
    )
    parser.add_argument(
        "--opt_dir",
        type=str,
        default="D:/Flood-Segmentation/dataset/HandLabeled/S2Hand",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="D:/Flood-Segmentation/dataset/HandLabeled/LabelHand",
    )
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--pos_weight", type=float, default=5.0)
    parser.add_argument("--out_dir", type=str, default="runs_flood")
    args = parser.parse_args()
    train(args)
