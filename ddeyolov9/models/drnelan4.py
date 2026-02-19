"""
DRNELAN4 Module - Replaces RepNCSPELAN4 in YOLOv9
Uses DilatedReparamBlock to improve receptive field
"""
import torch
import torch.nn as nn
from .dilated_reparam import DilatedReparamBlock


class DRNCSP(nn.Module):
    """DRNCSP block using DilatedReparamBlock"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1)
        
        # Replace RepConvN with DilatedReparamBlock
        self.m = nn.Sequential(
            *[DRNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DRNBottleneck(nn.Module):
    """DRN Bottleneck using DilatedReparamBlock"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # Use DilatedReparamBlock instead of RepConvN
        self.cv1 = DilatedReparamBlock(c1, c_, k=9, kernel_sizes=[5, 3, 3, 3], dilations=[1, 2, 3, 4])
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv3(torch.cat((self.cv1(x), self.cv2(x)), dim=1)) if self.add else self.cv3(torch.cat((self.cv1(x), self.cv2(x)), dim=1))


class DRNELAN4(nn.Module):
    """
    DRNELAN4 module - Enhanced version of RepNCSPELAN4
    Replaces RepNCSPELAN4 in YOLOv9 backbone
    """
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = nn.Conv2d(c1, c3, 1, 1)
        self.cv2 = nn.Conv2d(c1, c3, 1, 1)
        self.cv3 = nn.Conv2d(c3, c3, 1, 1)
        # Replace RepNCSP with DRNCSP
        self.cv4 = DRNCSP(c3, c3, n=n)
        self.cv5 = DRNCSP(c3, c3, n=n)
        self.cv6 = nn.Conv2d(c3, c4, 1, 1)
    
    def forward(self, x):
        y = torch.cat((self.cv1(x), self.cv2(x)), dim=1)
        return self.cv6(self.cv5(self.cv4(self.cv3(y))))

