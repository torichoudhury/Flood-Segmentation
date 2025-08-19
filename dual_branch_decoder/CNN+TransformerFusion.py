import torch
import torch.nn as nn
import timm

# ✅ CNN Decoder (U-Net Style)
class CNNDecoder(nn.Module):
    def __init__(self, input_channels=512):
        super(CNNDecoder, self).__init__()

        self.up1 = nn.ConvTranspose2d(input_channels, 256, kernel_size=2, stride=2)  # [B, 512, 16, 16] → [B, 256, 32, 32]
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # [B, 128, 32, 32] → [B, 64, 64, 64]
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


# ✅ Fixed Transformer Decoder - Using custom architecture instead of Swin
class TransformerDecoder(nn.Module):
    def __init__(self, input_channels=512):
        super(TransformerDecoder, self).__init__()

        # Instead of using pretrained Swin, create a custom upsampling path
        self.up1 = nn.ConvTranspose2d(input_channels, 256, kernel_size=2, stride=2)  # [B, 512, 16, 16] → [B, 256, 32, 32]
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # [B, 256, 32, 32] → [B, 256, 64, 64]
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
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C/8
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


# ✅ Dual-Branch Decoder (CNN + Transformer Fusion)
class DualBranchDecoder(nn.Module):
    def __init__(self):
        super(DualBranchDecoder, self).__init__()

        # CNN Branch
        self.cnn_decoder = CNNDecoder(input_channels=512)

        # Transformer Branch
        self.transformer_decoder = TransformerDecoder(input_channels=512)

        # Fusion Layer
        self.fusion_conv = nn.Conv2d(512, 256, kernel_size=1)  # Fuse CNN & Transformer features

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
        fused_features = torch.cat([cnn_output, transformer_output], dim=1)  # [B, 512, 64, 64]
        fused_features = self.fusion_conv(fused_features)  # [B, 256, 64, 64]

        # Final segmentation mask
        segmentation_mask = self.final_conv(fused_features)  # [B, 1, 64, 64]

        return segmentation_mask


# ✅ Testing Model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = DualBranchDecoder().to(device)

    # Dummy input tensor (batch size 4, channels 512, height 16, width 16)
    sample_input = torch.randn(4, 512, 16, 16).to(device)

    # Forward pass
    segmentation_mask = model(sample_input)
    print("Final Segmentation Mask Shape:", segmentation_mask.shape)  # Expected: [4, 1, 64, 64]