import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ✅ ASPP Module
class ASPP(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, dilation_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()

        # 1️⃣ Parallel Dilated Convolutions with Different Rates
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 1x1 Conv
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rates[1], dilation=dilation_rates[1])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rates[2], dilation=dilation_rates[2])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rates[3], dilation=dilation_rates[3])

        # 2️⃣ Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        # 3️⃣ Final 1x1 Convolution to Merge Features
        self.conv_final = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1)

        # BatchNorm & Activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply dilated convolutions
        x1 = self.relu(self.bn(self.conv1(x)))
        x2 = self.relu(self.bn(self.conv2(x)))
        x3 = self.relu(self.bn(self.conv3(x)))
        x4 = self.relu(self.bn(self.conv4(x)))

        # Apply Global Average Pooling
        x5 = self.global_avg_pool(x)
        x5 = self.conv_global(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate along channel dimension
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        # Final 1x1 Convolution
        x = self.conv_final(x)

        return x


# ✅ Integrating ASPP with Swin Transformer
class FloodSegmentationModel(nn.Module):
    def __init__(self):
        super(FloodSegmentationModel, self).__init__()

        # Load Swin Transformer Feature Extractor
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        # Reduce Swin output channels (768 → 512)
        self.channel_reduction = nn.Conv2d(768, 512, kernel_size=1)

        # Downsample Swin output back to 16×16
        self.downsample = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)

        # ASPP Module for Multi-Scale Feature Learning
        self.aspp = ASPP(in_channels=512, out_channels=512)

    def forward(self, x):
        # Convert grayscale SAR/Optical image to 3-channel for Swin Transformer
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract feature maps from Swin Transformer
        swin_features = self.swin(x)[-1]  # Output: [B, 7, 7, 768]

        # ✅ Fix Shape: Convert [B, 7, 7, 768] → [B, 768, 7, 7]
        swin_features = swin_features.permute(0, 3, 1, 2)

        # Reduce channels and resize back
        swin_features = self.channel_reduction(swin_features)  # [B, 512, 7, 7]
        swin_features = self.downsample(swin_features)  # [B, 512, 16, 16]

        # Apply ASPP for multi-scale feature learning
        enhanced_features = self.aspp(swin_features)

        return enhanced_features


# ✅ Testing Model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = FloodSegmentationModel().to(device)

    # Dummy input tensor (batch size 4)
    sample_input = torch.randn(4, 1, 512, 512).to(device)

    # Forward pass
    output_features = model(sample_input)
    print("✅ ASPP Output Shape:", output_features.shape)  # Expected: [4, 512, 16, 16]
