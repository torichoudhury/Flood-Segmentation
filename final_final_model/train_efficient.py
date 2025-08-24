"""
Efficient training script for flood segmentation with simplified architecture:
- Streamlined architecture for better efficiency
- Focus on core U-Net style decoder with self-attention

This file is self-contained and optimized for faster training and inference.

Requirements:
    pip install timm torch torchvision

Dataset expectation:
    Implement/adjust the FloodDataset to return (sar, optical, mask) where
      sar:     FloatTensor [1,H,W] in range ~[-1,1] or [0,1]
      optical: FloatTensor [1,H,W] (if you have RGB, adapt to 3 and modify code)
      mask:    FloatTensor [1,H,W] with 0/1 values
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
#  Lightweight Self-Attention
# =====================
class EfficientSelfAttention(nn.Module):
    """Simplified self-attention with reduced computational cost."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        inter = max(in_channels // reduction, 16)
        self.query = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.key = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Downsample for efficiency if spatial size is large
        if h * w > 1024:  # 32x32
            x_down = F.adaptive_avg_pool2d(x, (16, 16))
            q = self.query(x_down).view(b, -1, 256).permute(0, 2, 1)  # [B, 256, C']
            k = self.key(x_down).view(b, -1, 256)  # [B, C', 256]
            attn = F.softmax(torch.bmm(q, k), dim=-1)  # [B, 256, 256]
            v = self.value(x).view(b, c, h * w)  # [B, C, HW]
            # Upsample attention map
            attn_up = F.interpolate(
                attn.view(b, 16, 16, 16, 16).mean(dim=3),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).view(b, 256, h * w)
            out = torch.bmm(v, attn_up.permute(0, 2, 1)).view(b, c, h, w)
        else:
            q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)  # [B, HW, C']
            k = self.key(x).view(b, -1, h * w)  # [B, C', HW]
            attn = F.softmax(torch.bmm(q, k), dim=-1)  # [B, HW, HW]
            v = self.value(x).view(b, c, h * w)  # [B, C, HW]
            out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)

        return self.norm(self.gamma * out + x)


# =====================
#  Simplified Decoders
# =====================
class SimpleUpBlock(nn.Module):
    """Lightweight upsampling block."""

    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attention = (
            EfficientSelfAttention(out_ch, reduction=8) if use_attention else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        if self.attention:
            x = self.attention(x)
        return x


class EfficientCNNDecoder(nn.Module):
    """Streamlined CNN decoder without heavy attention mechanisms."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.up1 = SimpleUpBlock(in_channels, 256, use_attention=True)  # 1/16 -> 1/8
        self.up2 = SimpleUpBlock(256, 128, use_attention=False)  # 1/8  -> 1/4
        self.up3 = SimpleUpBlock(128, 64, use_attention=False)  # 1/4  -> 1/2
        self.up4 = SimpleUpBlock(64, 32, use_attention=False)  # 1/2  -> 1/1
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),  # Add dropout for regularization
            nn.Conv2d(16, 2, 1),  # 2 classes for CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out(x)


class EfficientDualBranchDecoder(nn.Module):
    """Simplified dual branch decoder with lightweight fusion."""

    def __init__(self, in_channels: int = 512):
        super().__init__()
        # Main CNN branch
        self.cnn = EfficientCNNDecoder(in_channels)

        # Lightweight transformer branch
        self.tf_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.tf_attn = EfficientSelfAttention(256, reduction=4)
        self.tf_up = nn.Sequential(
            SimpleUpBlock(256, 128),  # x2
            SimpleUpBlock(128, 64),  # x2
            SimpleUpBlock(64, 32),  # x2
            nn.Conv2d(32, 2, 1),  # 2 classes for CrossEntropyLoss
        )

        # Simple fusion - weighted average
        self.fusion_weight = nn.Parameter(
            torch.tensor(0.7)
        )  # Learnable weight for CNN branch

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),  # Input 2 channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),  # Output 2 classes
        )

    def forward(self, x):
        # CNN branch
        m_cnn = self.cnn(x)

        # Lightweight transformer branch
        tf_feat = self.tf_reduce(x)
        tf_feat = self.tf_attn(tf_feat)
        m_tf = self.tf_up(tf_feat)

        # Ensure same spatial size
        if m_cnn.shape[-2:] != m_tf.shape[-2:]:
            m_tf = F.interpolate(
                m_tf, size=m_cnn.shape[-2:], mode="bilinear", align_corners=False
            )

        # Simple weighted fusion
        w = torch.sigmoid(self.fusion_weight)
        fused = w * m_cnn + (1.0 - w) * m_tf

        # Final refinement
        out = self.refine(fused)
        return out


# =====================
#  Efficient Encoder
# =====================
class EfficientFeatureExtractor(nn.Module):
    """Simplified feature extractor without CBAM."""

    def __init__(self, model_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        # Use lighter backbone
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True
        )
        feat_ch = self.backbone.feature_info.channels()[-1]

        # Adaptive input layers
        self.sar_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.opt_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        # Initialize adaptors
        nn.init.kaiming_uniform_(self.sar_adapt.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.opt_adapt.weight, a=math.sqrt(5))

        # Simple fusion and reduction
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_ch * 2, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        # Extract features from both modalities
        s = self.backbone(self.sar_adapt(sar))[-1]
        o = self.backbone(self.opt_adapt(optical))[-1]

        # Simple concatenation and fusion
        x = torch.cat([s, o], dim=1)
        x = self.fuse(x)
        return x


class EfficientFloodSegmentationModel(nn.Module):
    """Streamlined flood segmentation model without ASPP/CBAM."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        use_dual_decoder: bool = True,
    ):
        super().__init__()
        self.encoder = EfficientFeatureExtractor(encoder_name, pretrained=True)
        self.decoder = (
            EfficientDualBranchDecoder(512)
            if use_dual_decoder
            else EfficientCNNDecoder(512)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use more conservative initialization
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # Scale down the weights slightly to prevent overconfident predictions
                with torch.no_grad():
                    m.weight *= 0.8
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Parameter):
                # Initialize learnable parameters like fusion_weight
                if "fusion_weight" in str(m):
                    nn.init.constant_(m, 0.6)  # Slight bias towards CNN branch
                else:
                    nn.init.xavier_uniform_(m)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        x = self.encoder(sar, optical)
        x = self.decoder(x)
        return x


# =====================
#  Loss & Metrics (Updated with Focal Loss and balanced weights)
# =====================
class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert 2-class prediction to probabilities
        if pred.shape[1] == 2:
            pred = F.softmax(pred, dim=1)[:, 1:2]  # Take flood class
        else:
            pred = torch.sigmoid(pred)

        # Flatten tensors using reshape instead of view
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        # Calculate dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined Focal Loss and Dice Loss."""

    def __init__(self, alpha=None, gamma=2.0, dice_weight=0.3):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return focal + self.dice_weight * dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert target from [B,1,H,W] float to [B,H,W] long for CrossEntropy
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        target = target.long()

        # Handle single channel predictions
        if pred.shape[1] == 1:
            # Convert single-channel output to 2-class output
            background_logits = -pred
            pred = torch.cat([background_logits, pred], dim=1)

        # Standard cross entropy
        ce_loss = F.cross_entropy(
            pred, target, reduction="none", ignore_index=self.ignore_index
        )

        # Calculate pt
        pt = torch.exp(-ce_loss)

        # Apply focal loss formula
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            # More balanced weights: [background: 1, flood: 3]
            class_weights = torch.tensor([1.0, 3.0])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert target from [B,1,H,W] float to [B,H,W] long for CrossEntropy
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        target = target.long()

        # Convert predictions to proper format for CrossEntropy
        # pred should be [B, num_classes, H, W] where num_classes=2
        if pred.shape[1] == 1:
            # Convert single-channel output to 2-class output
            # Assume pred is logits for flood class
            background_logits = -pred  # Background is inverse of flood
            pred = torch.cat([background_logits, pred], dim=1)

        return self.criterion(pred, target)


def batch_metrics(
    pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5
) -> Dict[str, float]:
    # Handle 2-class output from CrossEntropyLoss
    if pred.shape[1] == 2:
        # Apply softmax to get probabilities, take flood class (class 1)
        pred_prob = F.softmax(pred, dim=1)[:, 1:2]  # Keep [B,1,H,W] shape
    else:
        pred_prob = torch.sigmoid(pred)

    # Use adaptive threshold based on prediction statistics
    # This helps when predictions are too confident
    pred_flat_all = pred_prob.reshape(-1)
    target_flat_all = target.reshape(-1)

    # Find optimal threshold using Youden's J statistic (balanced accuracy)
    thresholds = torch.linspace(0.1, 0.9, 17).to(pred.device)
    best_f1 = 0.0
    best_threshold = thr

    for test_thr in thresholds:
        pred_test = (pred_prob > test_thr).float()
        pred_test_flat = pred_test.reshape(-1)

        tp = ((pred_test_flat == 1) & (target_flat_all == 1)).sum().float()
        fp = ((pred_test_flat == 1) & (target_flat_all == 0)).sum().float()
        fn = ((pred_test_flat == 0) & (target_flat_all == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = test_thr

    # Use the best threshold found
    pred_bin = (pred_prob > best_threshold).float()

    # Flatten tensors for easier computation
    pred_flat = pred_bin.reshape(-1)
    target_flat = target.reshape(-1)

    # Calculate metrics
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().float()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().float()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().float()

    # Calculate precision, recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Calculate IoU (Jaccard Index)
    iou = tp / (tp + fp + fn + 1e-8)

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "best_threshold": float(best_threshold),
    }


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
#  Train / Validate (Optimized)
# =====================
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    out_dir: str,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    tot_loss, n = 0.0, 0
    all_preds, all_targets = [], []

    for sar, opt, m in tqdm(loader, desc="Validating", leave=False):
        sar, opt, m = (
            sar.to(device, non_blocking=True),
            opt.to(device, non_blocking=True),
            m.to(device, non_blocking=True),
        )

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(sar, opt)
            if pred.shape[-2:] != m.shape[-2:]:
                pred = F.interpolate(
                    pred, size=m.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = criterion(pred, m)

        tot_loss += float(loss.item())
        n += 1

        # Store predictions and targets for overall metric calculation
        # Convert 2-class output to single channel probabilities
        if pred.shape[1] == 2:
            pred_prob = F.softmax(pred, dim=1)[:, 1:2]  # Take flood class probability
        else:
            pred_prob = torch.sigmoid(pred)
        all_preds.append(pred_prob.cpu())
        all_targets.append(m.cpu())

    # Calculate metrics over the entire validation set
    val_metrics = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
    }
    if all_preds:
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        # Use the improved batch_metrics function
        val_metrics = batch_metrics(all_preds_tensor, all_targets_tensor, thr=0.5)

    # Save validation visualization (less frequent for efficiency)
    if epoch % 5 == 0:
        os.makedirs(out_dir, exist_ok=True)
        try:
            sar0, opt0, m0 = next(iter(loader))
            sar0, opt0 = sar0[:2].to(device), opt0[:2].to(device)  # Only 2 samples
            with torch.no_grad():
                pr0 = model(sar0, opt0)
                # Convert 2-class output to single channel for visualization
                if pr0.shape[1] == 2:
                    pr0 = F.softmax(pr0, dim=1)[:, 1:2]  # Take flood class
                else:
                    pr0 = torch.sigmoid(pr0)
                grid = torch.cat(
                    [sar0.cpu(), opt0.cpu(), m0[:2].cpu(), pr0.cpu()], dim=0
                )
                save_image(
                    grid, os.path.join(out_dir, f"val_epoch{epoch:03d}.png"), nrow=2
                )
        except Exception:
            pass

    return tot_loss / max(n, 1), val_metrics


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
        num_workers=0,  # Reduced to avoid Windows multiprocessing issues
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    # Model - using lighter backbone for efficiency
    model = EfficientFloodSegmentationModel(
        encoder_name=args.encoder,
        use_dual_decoder=args.use_dual_decoder,
    ).to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss & Optimizer with Combined Loss for better class balance
    # Based on actual flood ratio of ~14.7%,
    alpha_weights = torch.tensor([0.85, 0.15]).float()  # Proportional to class distribution
    
    if device.type == "cuda":
        alpha_weights = alpha_weights.cuda()

    # Use Combined Loss (Focal + Dice) with adjusted weights
    criterion = CombinedLoss(alpha=alpha_weights, gamma=2.0, dice_weight=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )  # CosineAnnealingWarmRestarts scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        len(train_loader) * 10,  # T_0: restart every 10 epochs
        T_mult=2,  # Double the restart period each time
        eta_min=0,  # Minimum learning rate
        last_epoch=-1,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Train loop
    best_f1 = -1.0
    patience_counter = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (sar, opt, m) in enumerate(loop):
            sar, opt, m = (
                sar.to(device, non_blocking=True),
                opt.to(device, non_blocking=True),
                m.to(device, non_blocking=True),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(sar, opt)
                if pred.shape[-2:] != m.shape[-2:]:
                    pred = F.interpolate(
                        pred, size=m.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss = criterion(pred, m)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Step scheduler after each batch for CosineAnnealingWarmRestarts

            running_loss += loss.item()
            loop.set_postfix(
                loss=running_loss / (batch_idx + 1), lr=optimizer.param_groups[0]["lr"]
            )

        # Validation
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, args.out_dir, epoch
        )
        val_f1 = val_metrics["f1"]

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {running_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Best Thr: {val_metrics.get('best_threshold', 0.5):.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # No need to step scheduler here - it's stepped after each batch for CosineAnnealingWarmRestarts        # Save best model and early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                },
                os.path.join(args.out_dir, "best_model.pth"),
            )
            print(f"✅ Saved new best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break

    print(f"Training complete. Best F1: {best_f1:.4f}")


# =====================
#  CLI
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Efficient Flood Segmentation Training"
    )
    parser.add_argument(
        "--use_external_dataset",
        action="store_true",
        help="Use your dataset.FloodDataset instead of placeholder",
    )
    parser.add_argument(
        "--sar_dir",
        type=str,
        default="D:/Flood-Segmentation/dataset/HandLabeled/S1Hand",
        help="Directory containing SAR images",
    )
    parser.add_argument(
        "--opt_dir",
        type=str,
        default="D:/Flood-Segmentation/dataset/HandLabeled/S2Hand",
        help="Directory containing optical images",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="D:/Flood-Segmentation/dataset/HandLabeled/LabelHand",
        help="Directory containing mask labels",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b1",
        ],
        help="Encoder architecture",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--val_split", type=float, default=0.15, help="Validation split ratio"
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=5.0,
        help="Positive class weight for BCE loss",
    )
    parser.add_argument(
        "--out_dir", type=str, default="runs_efficient", help="Output directory"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_dual_decoder",
        action="store_true",
        default=True,
        help="Use dual branch decoder",
    )

    args = parser.parse_args()
    train(args)
