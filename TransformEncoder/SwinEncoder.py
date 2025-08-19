import torch
import torch.nn as nn
import timm

# ✅ Transformer Encoder (Fixed for Feature Maps)
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=4):
        super(TransformerEncoder, self).__init__()

        # ✅ Load Swin Transformer for feature extraction
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",  
            pretrained=True,
            features_only=True  # ✅ Extract feature maps, not classification logits
        )

        # ✅ Convert feature maps to 3-channel input for Swin Transformer
        self.input_conv = nn.Conv2d(512, 3, kernel_size=1)  # Reduce from 512 → 3

        # ✅ Reduce Swin output channels from 768 → 512
        self.channel_reduction = nn.Conv2d(768, 512, kernel_size=1)

        # ✅ Downsample Swin Transformer output back to 16×16
        self.downsample = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)

    def forward(self, features):
        # ✅ Convert feature map (512 channels) to Swin-compatible 3-channel input
        features = self.input_conv(features)  # [B, 512, 16, 16] → [B, 3, 16, 16]

        # ✅ Upsample to Swin Transformer input size (224×224)
        features = nn.functional.interpolate(features, size=(224, 224), mode="bilinear", align_corners=False)

        # ✅ Extract feature maps from Swin Transformer
        refined_features = self.swin(features)[-1]  # ✅ Extract last feature map

        # ✅ Ensure correct shape: [B, C, H, W]
        refined_features = refined_features.permute(0, 3, 1, 2)  # [B, 7, 7, 768] → [B, 768, 7, 7]

        # ✅ Reduce channels & resize back to 16×16
        refined_features = self.channel_reduction(refined_features)  # [B, 512, 7, 7]
        refined_features = self.downsample(refined_features)  # [B, 512, 16, 16]

        return refined_features


# ✅ Integrating Transformer with EfficientNet
class FloodFeatureExtractorWithTransformer(nn.Module):
    def __init__(self):
        super(FloodFeatureExtractorWithTransformer, self).__init__()

        # EfficientNet Feature Extractor
        self.backbone = timm.create_model("tf_efficientnet_b4", pretrained=True, features_only=True)
        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze EfficientNet layers

        # Reduce feature map channels
        self.reduce_channels = nn.Conv2d(448 * 2, 512, kernel_size=1)  # Reduce from 896 → 512

        # Transformer Encoder
        self.transformer = TransformerEncoder(embed_dim=512, num_heads=8, depth=4)

    def forward(self, sar, optical):
        # Convert 1-channel to 3-channel for EfficientNet
        sar = sar.repeat(1, 3, 1, 1)  
        optical = optical.repeat(1, 3, 1, 1)

        # Extract features
        sar_features = self.backbone(sar)[-1]  
        optical_features = self.backbone(optical)[-1]

        # Merge SAR & Optical features
        combined_features = torch.cat([sar_features, optical_features], dim=1)  # [B, 896, 16, 16]
        combined_features = self.reduce_channels(combined_features)  # [B, 512, 16, 16]

        # Pass through Transformer Encoder (Fixed)
        refined_features = self.transformer(combined_features)  # ✅ Output: [B, 512, 16, 16]
        return refined_features


# ✅ Testing Model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = FloodFeatureExtractorWithTransformer().to(device)

    # Dummy input tensors (batch size 4)
    sar_input = torch.randn(4, 1, 512, 512).to(device)
    optical_input = torch.randn(4, 1, 512, 512).to(device)

    # Forward pass
    refined_features = model(sar_input, optical_input)
    print("Transformer Refined Feature Shape:", refined_features.shape)  # Expected: [4, 512, 16, 16]
