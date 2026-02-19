"""
DCNv4-Dyhead Detection Head
Combines DCNv4 with Dynamic Head for multi-scale feature learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcnv4 import DCNv4


class ScaleAwareAttention(nn.Module):
    """
    Scale-aware attention module πL
    Dynamically integrates features based on semantic importance of different scales
    """
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, F):
        """
        F: [L, S, C] where L=levels, S=spatial, C=channels
        """
        L, S, C = F.shape[0], F.shape[1] * F.shape[2], F.shape[3]
        
        # Reshape to [B, C, H, W] for conv
        B = F.shape[0]
        F_reshaped = F.view(B, C, F.shape[1], F.shape[2])
        
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(F_reshaped, 1)  # [B, C, 1, 1]
        
        # 1x1 conv approximation
        scale_weight = self.fc(avg_pool)  # [B, C, 1, 1]
        
        # Hard sigmoid activation: σ(x) = max(0, min(1, (x+1)/2))
        scale_weight = torch.clamp((scale_weight + 1) / 2, 0, 1)
        
        # Apply attention
        out = F_reshaped * scale_weight
        
        return out.view(B, F.shape[1], F.shape[2], C)


class SpatialAwareAttention(nn.Module):
    """
    Spatial-aware attention module πS (Improved with DCNv4)
    Focuses on discriminative regions consistent between spatial positions
    """
    def __init__(self, channels, num_groups=4, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        
        # DCNv4 for spatial aggregation
        self.dcnv4 = DCNv4(channels, channels, kernel_size=kernel_size, groups=num_groups)
        
        # Normalization
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, F):
        """
        F: [B, H, W, C] or [B, C, H, W]
        """
        # Ensure [B, C, H, W] format
        if F.dim() == 4 and F.shape[-1] == self.channels:
            F = F.permute(0, 3, 1, 2)
        
        B, C, H, W = F.shape
        
        # DCNv4 spatial aggregation
        aggregated = self.dcnv4(F)  # [B, C, H, W]
        
        # Normalize
        aggregated = aggregated.permute(0, 2, 3, 1)  # [B, H, W, C]
        aggregated = self.norm(aggregated)
        aggregated = aggregated.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Sigmoid for attention
        attention = torch.sigmoid(aggregated)
        
        # Apply attention
        out = F * attention
        
        return out


class TaskAwareAttention(nn.Module):
    """
    Task-aware attention module πC
    Supports different downstream tasks (classification, regression)
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU()
        )
        self.beta1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU()
        )
        self.alpha2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU()
        )
        self.beta2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, F):
        """
        F: [B, C, H, W]
        """
        # Compute dynamic parameters
        alpha1 = self.alpha1(F)  # [B, C, 1, 1]
        beta1 = self.beta1(F)    # [B, C, 1, 1]
        alpha2 = self.alpha2(F)   # [B, C, 1, 1]
        beta2 = self.beta2(F)     # [B, C, 1, 1]
        
        # Apply task-aware transformation
        out1 = alpha1 * F + beta1
        out2 = alpha2 * F + beta2
        
        # Max operation
        out = torch.max(out1, out2)
        
        return out


class DCNv4Dyhead(nn.Module):
    """
    DCNv4-Dyhead: Combines scale-aware, spatial-aware (DCNv4), and task-aware attention
    """
    def __init__(self, channels, num_levels=3, num_groups=4):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        
        # Three attention modules
        self.scale_attention = ScaleAwareAttention(channels)
        self.spatial_attention = SpatialAwareAttention(channels, num_groups=num_groups)
        self.task_attention = TaskAwareAttention(channels)
    
    def forward(self, features):
        """
        features: List of feature maps from different levels [F1, F2, F3]
        Each F: [B, C, H, W]
        """
        # Stack features from different levels
        # For simplicity, we'll process each level separately and combine
        outputs = []
        
        for F in features:
            # Ensure proper format
            if F.dim() == 4:
                B, C, H, W = F.shape
                # Reshape for scale attention: [B, H, W, C]
                F_reshaped = F.permute(0, 2, 3, 1)
                
                # Apply scale-aware attention
                F = self.scale_attention(F_reshaped)
                F = F.permute(0, 3, 1, 2)  # Back to [B, C, H, W]
                
                # Apply spatial-aware attention (DCNv4)
                F = self.spatial_attention(F)
                
                # Apply task-aware attention
                F = self.task_attention(F)
                
                outputs.append(F)
        
        return outputs

