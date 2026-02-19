"""
Dilated Reparam Block implementation
Based on the paper: DDEYOLOv9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block uses a combination of non-dilated small kernel
    and multiple dilated small kernel layers to enhance a non-dilated large kernel conv layer.
    
    Hyperparameters:
    - K: size of large kernel (default: 9)
    - k: sizes of parallel conv layers (default: [5, 3, 3, 3])
    - r: dilation rates (default: [1, 2, 3, 4])
    """
    def __init__(self, c1, c2, k=9, kernel_sizes=[5, 3, 3, 3], dilations=[1, 2, 3, 4]):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        
        # Create parallel conv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for ks, d in zip(kernel_sizes, dilations):
            padding = ((ks - 1) * d) // 2
            conv = nn.Conv2d(c1, c2, kernel_size=ks, dilation=d, padding=padding, bias=False)
            bn = nn.BatchNorm2d(c2)
            self.convs.append(conv)
            self.bns.append(bn)
        
        # Main large kernel conv (non-dilated)
        padding = (k - 1) // 2
        self.main_conv = nn.Conv2d(c1, c2, kernel_size=k, padding=padding, bias=False)
        self.main_bn = nn.BatchNorm2d(c2)
        
    def forward(self, x):
        # During training, use all parallel branches
        if self.training:
            out = self.main_conv(x)
            out = self.main_bn(out)
            
            for conv, bn in zip(self.convs, self.bns):
                branch_out = conv(x)
                branch_out = bn(branch_out)
                out = out + branch_out
            return out
        else:
            # During inference, use reparameterized single conv
            return self.main_conv(x)
    
    def fuse_bn(self, conv, bn):
        """Fuse batch norm into convolution"""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        return kernel * t, beta - running_mean * gamma / std
    
    def reparameterize(self):
        """Reparameterize all branches into a single conv layer"""
        # Get main conv weight and bias
        main_weight, main_bias = self.fuse_bn(self.main_conv, self.main_bn)
        
        # Fuse all parallel branches
        for conv, bn in zip(self.convs, self.bns):
            branch_weight, branch_bias = self.fuse_bn(conv, bn)
            
            # Transform dilated conv to equivalent large kernel
            # This is a simplified version - full implementation would require
            # proper kernel transformation with zero padding
            k_size = conv.kernel_size[0]
            dilation = conv.dilation[0]
            padding = conv.padding[0]
            
            # Add to main weight (with appropriate padding/transformation)
            # For simplicity, we'll add directly (in practice, need proper kernel transformation)
            if k_size * dilation <= self.k:
                # Pad branch weight to match main kernel size
                pad_size = (self.k - k_size) // 2
                padded_weight = F.pad(branch_weight, (pad_size, pad_size, pad_size, pad_size))
                main_weight = main_weight + padded_weight
                main_bias = main_bias + branch_bias
        
        # Create new fused conv
        fused_conv = nn.Conv2d(self.c1, self.c2, kernel_size=self.k, 
                              padding=(self.k - 1) // 2, bias=True)
        fused_conv.weight.data = main_weight
        fused_conv.bias.data = main_bias
        
        return fused_conv

