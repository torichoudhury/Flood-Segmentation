import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp


# CBAM Module (Channel + Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        # Channel Attention (Global Avg & Max Pooling)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False
            ),
            nn.Sigmoid(),
        )

        # Spatial Attention (Convolution-Based Attention)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        channel_att_map = self.channel_att(x)
        x = x * channel_att_map  # Element-wise multiplication

        # Spatial Attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att_map = self.spatial_att(spatial_input)
        x = x * spatial_att_map  # Element-wise multiplication

        return x


# ASPP Module
class ASPP(nn.Module):
    def __init__(
        self, in_channels=512, out_channels=512, dilation_rates=[1, 6, 12, 18]
    ):
        super(ASPP, self).__init__()

        # Parallel Dilated Convolutions with Different Rates
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )  # 1x1 Conv
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation_rates[1],
            dilation=dilation_rates[1],
        )
        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation_rates[2],
            dilation=dilation_rates[2],
        )
        self.conv4 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation_rates[3],
            dilation=dilation_rates[3],
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        # Final 1x1 Convolution to Merge Features
        self.conv_final = nn.Conv2d(
            out_channels * 5, out_channels, kernel_size=1, stride=1
        )

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


# ASPP + CBAM Integration
class ASPP_CBAM(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super(ASPP_CBAM, self).__init__()

        # ASPP for multi-scale feature extraction
        self.aspp = ASPP(in_channels=in_channels, out_channels=out_channels)

        # CBAM for attention refinement
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.aspp(x)  # Multi-scale feature extraction
        x = self.cbam(x)  # Apply attention
        return x


# Transformer Encoder using Swin Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=4):
        super(TransformerEncoder, self).__init__()

        # Load Swin Transformer for feature extraction
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,  # Extract feature maps, not classification logits
        )

        # Convert feature maps to 3-channel input for Swin Transformer
        self.input_conv = nn.Conv2d(512, 3, kernel_size=1)  # Reduce from 512 → 3

        # Reduce Swin output channels from 768 → 512
        self.channel_reduction = nn.Conv2d(768, 512, kernel_size=1)

        # Downsample Swin Transformer output back to 16×16
        self.downsample = nn.Upsample(
            size=(16, 16), mode="bilinear", align_corners=False
        )

    def forward(self, features):
        # Convert feature map (512 channels) to Swin-compatible 3-channel input
        features = self.input_conv(features)  # [B, 512, 16, 16] → [B, 3, 16, 16]

        # Upsample to Swin Transformer input size (224×224)
        features = nn.functional.interpolate(
            features, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Extract feature maps from Swin Transformer
        refined_features = self.swin(features)[-1]  # Extract last feature map

        # Ensure correct shape: [B, C, H, W]
        refined_features = refined_features.permute(
            0, 3, 1, 2
        )  # [B, 7, 7, 768] → [B, 768, 7, 7]

        # Reduce channels & resize back to 16×16
        refined_features = self.channel_reduction(refined_features)  # [B, 512, 7, 7]
        refined_features = self.downsample(refined_features)  # [B, 512, 16, 16]

        return refined_features


# Simple self-attention module for capturing global context
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()

        # Reshape and transpose for attention calculation
        proj_query = (
            self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        )  # B x HW x C/8
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C/8 x HW

        # Calculate attention map
        attention = torch.bmm(proj_query, proj_key)  # B x HW x HW
        attention = self.softmax(attention)

        # Apply attention to values
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, height, width)

        # Residual connection with learnable parameter
        out = self.gamma * out + x
        return out


# CNN Decoder (U-Net Style)
class CNNDecoder(nn.Module):
    def __init__(self, input_channels=512):
        super(CNNDecoder, self).__init__()

        self.up1 = nn.ConvTranspose2d(
            input_channels, 256, kernel_size=2, stride=2
        )  # [B, 512, 16, 16] → [B, 256, 32, 32]
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )  # [B, 128, 32, 32] → [B, 64, 64, 64]
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)  # Output segmentation mask

    def forward(self, x):
        x = self.up1(x)
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.up2(x)
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.final_conv(x)
        return x  # Output shape: [B, 1, 64, 64]


# Custom Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, input_channels=512):
        super(TransformerDecoder, self).__init__()

        # Custom upsampling path
        self.up1 = nn.ConvTranspose2d(
            input_channels, 256, kernel_size=2, stride=2
        )  # [B, 512, 16, 16] → [B, 256, 32, 32]
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose2d(
            256, 256, kernel_size=2, stride=2
        )  # [B, 256, 32, 32] → [B, 256, 64, 64]
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        # Self-attention module for global context
        self.attention = SelfAttention(256)

    def forward(self, x):
        # Upsampling path
        x = self.up1(x)
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.up2(x)
        x = self.relu(self.bn2(self.conv2(x)))

        # Apply self-attention for global context
        x = self.attention(x)

        return x  # Output shape: [B, 256, 64, 64]


# Dual-Branch Decoder (CNN + Transformer Fusion)
class DualBranchDecoder(nn.Module):
    def __init__(self):
        super(DualBranchDecoder, self).__init__()

        # CNN Branch
        self.cnn_decoder = CNNDecoder(input_channels=512)

        # Transformer Branch
        self.transformer_decoder = TransformerDecoder(input_channels=512)

        # Fusion Layer
        self.fusion_conv = nn.Conv2d(
            512, 256, kernel_size=1
        )  # Fuse CNN & Transformer features

        # Final segmentation head
        self.final_conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        # CNN Decoder Output (Local Features)
        cnn_output = self.cnn_decoder(x)  # [B, 1, 64, 64]

        # Transformer Decoder Output (Global Features)
        transformer_output = self.transformer_decoder(x)  # [B, 256, 64, 64]

        # Expand CNN output to match transformer channels
        cnn_output = cnn_output.expand(-1, 256, -1, -1)  # [B, 256, 64, 64]

        # Fusion: Concatenate and pass through 1x1 conv
        fused_features = torch.cat(
            [cnn_output, transformer_output], dim=1
        )  # [B, 512, 64, 64]
        fused_features = self.fusion_conv(fused_features)  # [B, 256, 64, 64]

        # Final segmentation mask
        segmentation_mask = self.final_conv(fused_features)  # [B, 1, 64, 64]

        return segmentation_mask


# Main feature extractor with transformer
class FloodFeatureExtractorWithTransformer(nn.Module):
    def __init__(self):
        super(FloodFeatureExtractorWithTransformer, self).__init__()

        # EfficientNet Feature Extractor
        self.backbone = timm.create_model(
            "tf_efficientnet_b4", pretrained=True, features_only=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze EfficientNet layers

        # Reduce feature map channels
        self.reduce_channels = nn.Conv2d(
            448 * 2, 512, kernel_size=1
        )  # Reduce from 896 → 512

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
        combined_features = torch.cat(
            [sar_features, optical_features], dim=1
        )  # [B, 896, 16, 16]
        combined_features = self.reduce_channels(combined_features)  # [B, 512, 16, 16]

        # Pass through Transformer Encoder
        refined_features = self.transformer(
            combined_features
        )  # Output: [B, 512, 16, 16]
        return refined_features


# Complete Flood Segmentation Model (Simplified)
class FloodSegmentationModel(nn.Module):
    def __init__(self):
        super(FloodSegmentationModel, self).__init__()

        # Use a simpler, proven architecture like DeepLabV3+
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",  # simpler backbone
            encoder_weights="imagenet",
            in_channels=2,  # 1 for SAR + 1 for optical
            classes=1,
            activation=None,  # We'll apply sigmoid manually
        )

        # Better initialization for the final segmentation head
        for m in self.model.segmentation_head.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization with a smaller gain for better gradient flow
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # Initialize bias to a small negative value to make the model
                    # initially predict fewer positive pixels (helps with class imbalance)
                    nn.init.constant_(m.bias, -2.0)

        # Add dropout to prevent overfitting
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, sar, optical):
        # Combine channels
        x = torch.cat([sar, optical], dim=1)

        # Apply dropout during training
        if self.training:
            x = self.dropout(x)

        # Get segmentation output
        output = self.model(x)

        # Return single output (remove the dual output)
        return output


# For testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the complete model
    model = FloodSegmentationModel().to(device)

    # Test with dummy input
    sar_input = torch.randn(4, 1, 512, 512).to(device)
    optical_input = torch.randn(4, 1, 512, 512).to(device)

    # Forward pass
    segmentation_mask, encoder_output = model(sar_input, optical_input)
    print(
        "Segmentation Mask Shape:", segmentation_mask.shape
    )  # Expected: [4, 1, 64, 64]
    print("Encoder Output Shape:", encoder_output.shape)  # Expected: [4, 1, 64, 64]
