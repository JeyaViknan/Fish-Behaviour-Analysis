"""
DCNv4 (Deformable Convolution v4) implementation
Efficient dynamic sparse operator with adaptive aggregation windows
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNv4(nn.Module):
    """
    Deformable Convolution v4
    Uses adaptive aggregation windows and dynamic aggregation weights
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Offset and mask prediction
        self.offset_conv = nn.Conv2d(
            in_channels, 
            groups * kernel_size * kernel_size * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Modulation scalar (unbounded, no softmax)
        self.modulator_conv = nn.Conv2d(
            in_channels,
            groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Weight projection
        self.weight_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.modulator_conv.weight, 0)
        nn.init.constant_(self.modulator_conv.bias, 1.0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate offsets
        offset = self.offset_conv(x)  # [B, groups*K*K*2, H, W]
        offset = offset.view(B, self.groups, self.kernel_size * self.kernel_size, 2, H, W)
        
        # Generate modulation scalars (unbounded)
        modulator = self.modulator_conv(x)  # [B, groups*K*K, H, W]
        modulator = modulator.view(B, self.groups, self.kernel_size * self.kernel_size, H, W)
        
        # Generate sampling positions
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        coords = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
        coords = coords.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, 1, 1, 1, H, W]
        
        # Apply offsets
        sample_coords = coords + offset  # [B, groups, K*K, 2, H, W]
        
        # Normalize to [-1, 1]
        sample_coords[:, :, :, 0, :, :] = 2.0 * sample_coords[:, :, :, 0, :, :] / (W - 1) - 1.0
        sample_coords[:, :, :, 1, :, :] = 2.0 * sample_coords[:, :, :, 1, :, :] / (H - 1) - 1.0
        
        # Reshape for grid_sample
        sample_coords = sample_coords.permute(0, 1, 4, 5, 2, 3).contiguous()  # [B, groups, H, W, K*K, 2]
        sample_coords = sample_coords.view(B * self.groups, H, W, self.kernel_size * self.kernel_size, 2)
        
        # Sample features
        x_grouped = x.view(B * self.groups, C // self.groups, H, W)
        sampled_features = []
        
        for k in range(self.kernel_size * self.kernel_size):
            coords_k = sample_coords[:, :, :, k, :].unsqueeze(1)  # [B*groups, 1, H, W, 2]
            sampled = F.grid_sample(
                x_grouped,
                coords_k,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            sampled_features.append(sampled)
        
        sampled_features = torch.stack(sampled_features, dim=2)  # [B*groups, C//groups, K*K, H, W]
        
        # Apply modulation (unbounded weights)
        modulator = modulator.view(B * self.groups, self.kernel_size * self.kernel_size, H, W)
        modulator = modulator.unsqueeze(1)  # [B*groups, 1, K*K, H, W]
        
        # Aggregate
        aggregated = (sampled_features * modulator).sum(dim=2)  # [B*groups, C//groups, H, W]
        aggregated = aggregated.view(B, C, H, W)
        
        # Final projection
        out = self.weight_conv(aggregated)
        
        return out

