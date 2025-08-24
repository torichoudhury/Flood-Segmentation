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


# Improved CNN Decoder (U-Net Style with higher resolution)
class ImprovedCNNDecoder(nn.Module):
    def __init__(self, input_channels=512):
        super(ImprovedCNNDecoder, self).__init__()

        # More gradual upsampling for better quality
        self.up1 = nn.ConvTranspose2d(
            input_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # [B, 512, 16, 16] → [B, 256, 32, 32]
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # [B, 128, 32, 32] → [B, 64, 64, 64]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # [B, 32, 64, 64] → [B, 16, 128, 128]
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Final upsampling to match input resolution
        self.up4 = nn.ConvTranspose2d(
            16, 8, kernel_size=3, stride=4, padding=1, output_padding=3
        )  # [B, 16, 128, 128] → [B, 8, 512, 512]

        self.final_conv = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),  # Final segmentation
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.final_conv(x)

        return x  # Output shape: [B, 1, 512, 512]


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


# Improved Dual-Branch Decoder (CNN + Transformer Fusion)
class ImprovedDualBranchDecoder(nn.Module):
    def __init__(self):
        super(ImprovedDualBranchDecoder, self).__init__()

        # CNN Branch - produces high resolution output
        self.cnn_decoder = ImprovedCNNDecoder(input_channels=512)

        # Transformer Branch - produces feature maps
        self.transformer_decoder = TransformerDecoder(input_channels=512)

        # Upsample transformer features to match CNN output
        self.transformer_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 64→128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 128→256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 256→512
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Final channel reduction
        )

        # Feature fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # Combine 2 channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),  # Final segmentation
        )

        # Attention for feature fusion
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        # CNN Decoder Output (Local Features)
        cnn_output = self.cnn_decoder(x)  # [B, 1, 512, 512]

        # Transformer Decoder Output (Global Features)
        transformer_features = self.transformer_decoder(x)  # [B, 256, 64, 64]
        transformer_output = self.transformer_upsample(
            transformer_features
        )  # [B, 1, 512, 512]

        # Feature fusion with attention
        combined = torch.cat(
            [cnn_output, transformer_output], dim=1
        )  # [B, 2, 512, 512]
        attention_weights = self.fusion_attention(combined)  # [B, 1, 512, 512]

        # Apply attention weighting
        weighted_features = combined * attention_weights

        # Final fusion
        segmentation_mask = self.fusion_conv(weighted_features)  # [B, 1, 512, 512]

        return segmentation_mask


# Enhanced ResNet Feature Extractor
class AdvancedResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50"):
        super(AdvancedResNetFeatureExtractor, self).__init__()

        # ResNet Feature Extractor
        self.backbone = timm.create_model(
            model_name, pretrained=True, features_only=True
        )

        # Make more layers trainable for better fine-tuning
        total_params = len(list(self.backbone.parameters()))
        frozen_params = max(total_params - 80, 0)  # Freeze fewer layers

        for i, param in enumerate(self.backbone.parameters()):
            if i < frozen_params:  # Freeze early layers
                param.requires_grad = False
            else:
                param.requires_grad = True  # Fine-tune more layers

        # Channel adaptation layers with better initialization
        self.sar_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.optical_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        # Initialize adaptation layers
        nn.init.xavier_uniform_(self.sar_adapt.weight)
        nn.init.xavier_uniform_(self.optical_adapt.weight)

        # Feature fusion and channel reduction
        # ResNet-50 last layer has 2048 channels
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(2048 * 2, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Add CBAM attention after fusion
        self.attention = CBAM(512)

        # Final processing
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, sar, optical):
        # Convert 1-channel to 3-channel for ResNet
        sar_3ch = self.sar_adapt(sar)  # [B, 1, H, W] → [B, 3, H, W]
        optical_3ch = self.optical_adapt(optical)  # [B, 1, H, W] → [B, 3, H, W]

        # Extract features using ResNet
        sar_features = self.backbone(sar_3ch)[-1]  # Last feature map
        optical_features = self.backbone(optical_3ch)[-1]  # Last feature map

        # Merge SAR & Optical features
        combined_features = torch.cat(
            [sar_features, optical_features], dim=1
        )  # [B, 4096, H, W]

        # Reduce channels with better processing
        reduced_features = self.reduce_channels(combined_features)  # [B, 512, H, W]

        # Apply attention mechanism
        attended_features = self.attention(reduced_features)

        # Apply dropout
        output = self.dropout(attended_features)

        return output


# Advanced Flood Segmentation Model using all features
class AdvancedFloodSegmentationModel(nn.Module):
    def __init__(self, use_aspp=True, use_dual_decoder=True):
        super(AdvancedFloodSegmentationModel, self).__init__()

        self.use_aspp = use_aspp
        self.use_dual_decoder = use_dual_decoder

        # Advanced Feature Extractor
        self.feature_extractor = AdvancedResNetFeatureExtractor()

        # Multi-scale feature processing with ASPP + CBAM
        if use_aspp:
            self.multi_scale_processor = ASPP_CBAM(in_channels=512, out_channels=512)
        else:
            self.multi_scale_processor = nn.Identity()

        # Decoder selection
        if use_dual_decoder:
            # Use improved dual-branch decoder
            self.decoder = ImprovedDualBranchDecoder()
        else:
            # Use improved CNN decoder
            self.decoder = ImprovedCNNDecoder(input_channels=512)

        # Final output processing
        self.final_dropout = nn.Dropout2d(p=0.1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with proper initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # Initialize bias to small negative value for flood detection
                    nn.init.constant_(m.bias, -0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, sar, optical):
        # Extract advanced features from both modalities
        features = self.feature_extractor(sar, optical)  # [B, 512, H, W]

        # Multi-scale feature processing
        if self.use_aspp:
            features = self.multi_scale_processor(features)  # ASPP + CBAM processing

        # Decode features to segmentation mask
        output = self.decoder(features)

        # Apply final dropout during training
        if self.training:
            output = self.final_dropout(output)

        return output

    def get_feature_maps(self, sar, optical):
        """Return intermediate feature maps for visualization"""
        features = self.feature_extractor(sar, optical)

        if self.use_aspp:
            processed_features = self.multi_scale_processor(features)
        else:
            processed_features = features

        return {"raw_features": features, "processed_features": processed_features}


# Lightweight version for faster training/inference
class LightweightAdvancedFloodModel(nn.Module):
    def __init__(self):
        super(LightweightAdvancedFloodModel, self).__init__()

        # Use smaller ResNet backbone
        self.backbone = timm.create_model(
            "resnet34", pretrained=True, features_only=True
        )

        # Freeze most layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Channel adaptation
        self.adapt_conv = nn.Conv2d(2, 3, kernel_size=1)  # Combine both inputs

        # Lightweight ASPP
        self.aspp = ASPP(in_channels=512, out_channels=256)  # ResNet-34 features

        # Simple attention
        self.attention = CBAM(256)

        # Simple upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, sar, optical):
        # Combine inputs
        x = torch.cat([sar, optical], dim=1)  # [B, 2, H, W]
        x = self.adapt_conv(x)  # [B, 3, H, W]

        # Extract features
        features = self.backbone(x)[-1]  # Get last feature map

        # Process with ASPP and attention
        features = self.aspp(features)
        features = self.attention(features)

        # Decode
        output = self.decoder(features)

        return output


# For testing and comparison
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Test input
    sar_input = torch.randn(2, 1, 512, 512).to(device)
    optical_input = torch.randn(2, 1, 512, 512).to(device)

    # Test Advanced Model
    print("\n=== Testing Advanced Model ===")
    advanced_model = AdvancedFloodSegmentationModel(
        use_aspp=True, use_dual_decoder=True
    ).to(device)

    print(
        f"Advanced Model Parameters: {sum(p.numel() for p in advanced_model.parameters()):,}"
    )
    print(
        f"Trainable Parameters: {sum(p.numel() for p in advanced_model.parameters() if p.requires_grad):,}"
    )

    with torch.no_grad():
        output = advanced_model(sar_input, optical_input)
        print(f"Advanced Model Output Shape: {output.shape}")

    # Test Lightweight Model
    print("\n=== Testing Lightweight Model ===")
    lightweight_model = LightweightAdvancedFloodModel().to(device)

    print(
        f"Lightweight Model Parameters: {sum(p.numel() for p in lightweight_model.parameters()):,}"
    )
    print(
        f"Trainable Parameters: {sum(p.numel() for p in lightweight_model.parameters() if p.requires_grad):,}"
    )

    with torch.no_grad():
        output = lightweight_model(sar_input, optical_input)
        print(f"Lightweight Model Output Shape: {output.shape}")

    # Test feature extraction
    print("\n=== Testing Feature Extraction ===")
    with torch.no_grad():
        feature_maps = advanced_model.get_feature_maps(sar_input, optical_input)
        for key, value in feature_maps.items():
            print(f"{key}: {value.shape}")

    print("\n=== Model Comparison ===")
    # Simple model for comparison
    simple_model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=2,
        classes=1,
        activation=None,
    ).to(device)

    simple_params = sum(p.numel() for p in simple_model.parameters())
    advanced_params = sum(p.numel() for p in advanced_model.parameters())
    lightweight_params = sum(p.numel() for p in lightweight_model.parameters())

    print(f"Simple DeepLabV3+ Parameters: {simple_params:,}")
    print(f"Advanced Model Parameters: {advanced_params:,}")
    print(f"Lightweight Model Parameters: {lightweight_params:,}")
    print(f"Advanced/Simple Ratio: {advanced_params/simple_params:.2f}x")
    print(f"Lightweight/Simple Ratio: {lightweight_params/simple_params:.2f}x")
