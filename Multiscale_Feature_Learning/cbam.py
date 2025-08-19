import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ CBAM Module (Channel + Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        # **Channel Attention** (Global Avg & Max Pooling)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # **Spatial Attention** (Convolution-Based Attention)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # **Channel Attention**
        channel_att_map = self.channel_att(x)
        x = x * channel_att_map  # Element-wise multiplication

        # **Spatial Attention**
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att_map = self.spatial_att(spatial_input)
        x = x * spatial_att_map  # Element-wise multiplication

        return x

# ✅ Integrating CBAM with ASPP
class ASPP_CBAM(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super(ASPP_CBAM, self).__init__()

        # **Atrous Spatial Pyramid Pooling (ASPP)**
        self.aspp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=6, padding=6),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=12, padding=12),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=18, padding=18),
            nn.ReLU()
        )

        # **CBAM for Attention Refinement**
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.aspp(x)  # Multi-scale feature extraction
        x = self.cbam(x)  # Apply attention
        return x

# ✅ Testing the ASPP + CBAM Module
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ASPP + CBAM module
    model = ASPP_CBAM().to(device)

    # Dummy input tensor (batch size 4, channels 512, height 16, width 16)
    sample_input = torch.randn(4, 512, 16, 16).to(device)

    # Forward pass
    refined_output = model(sample_input)
    print("✅ CBAM-Refined ASPP Output Shape:", refined_output.shape)  # Expected: [4, 512, 16, 16]
