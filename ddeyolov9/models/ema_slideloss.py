"""
EMA-SlideLoss: Exponential Moving Average SlideLoss
Handles class imbalance by focusing on hard samples
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA_SlideLoss(nn.Module):
    """
    EMA-SlideLoss loss function
    Dynamically adjusts weights for easy/hard samples using exponential moving average
    """
    def __init__(self, decay=0.9, num_iterations=1000):
        super().__init__()
        self.decay = decay
        self.num_iterations = num_iterations
        self.register_buffer('ema_mu', torch.tensor(0.5))
        self.register_buffer('iteration', torch.tensor(0))
    
    def update_ema_mu(self, current_mu):
        """
        Update EMA of mu using Equation (6)
        EMAt = d * (1 - exp(-t/t_all)) * EMAt-1 + (1 - d) * auto_µ
        """
        self.iteration += 1
        t = self.iteration.float()
        t_all = self.num_iterations
        
        # Decay factor: d * (1 - exp(-t/t_all))
        decay_factor = self.decay * (1 - torch.exp(-t / t_all))
        
        # Update EMA
        self.ema_mu = decay_factor * self.ema_mu + (1 - self.decay) * current_mu
    
    def slide_weight(self, iou):
        """
        Calculate slide weight based on IoU and EMA mu
        Equation (7)
        """
        ema_mu = self.ema_mu.item()
        
        # Calculate weights based on IoU relative to EMA mu
        weights = torch.ones_like(iou)
        
        # x <= EMAt - 0.1: weight = 1
        mask1 = iou <= (ema_mu - 0.1)
        weights[mask1] = 1.0
        
        # EMAt - 0.1 < x < EMAt: weight = e^(1-EMAt)
        mask2 = (iou > (ema_mu - 0.1)) & (iou < ema_mu)
        weights[mask2] = torch.exp(torch.tensor(1.0 - ema_mu))
        
        # x >= EMAt: weight = e^(1-x)
        mask3 = iou >= ema_mu
        weights[mask3] = torch.exp(1.0 - iou[mask3])
        
        return weights
    
    def forward(self, pred_boxes, target_boxes, pred_scores, target_labels, 
                cls_loss_fn, reg_loss_fn):
        """
        Calculate EMA-SlideLoss
        
        Args:
            pred_boxes: Predicted bounding boxes [N, 4]
            target_boxes: Target bounding boxes [N, 4]
            pred_scores: Predicted class scores [N, num_classes]
            target_labels: Target class labels [N]
            cls_loss_fn: Classification loss function
            reg_loss_fn: Regression loss function
        
        Returns:
            Total loss
        """
        # Calculate IoU between predictions and targets
        ious = self.calculate_iou(pred_boxes, target_boxes)
        
        # Update EMA mu with current average IoU
        current_mu = ious.mean().item()
        self.update_ema_mu(torch.tensor(current_mu))
        
        # Calculate slide weights
        weights = self.slide_weight(ious)
        
        # Calculate classification loss
        cls_loss = cls_loss_fn(pred_scores, target_labels)
        
        # Calculate regression loss
        reg_loss = reg_loss_fn(pred_boxes, target_boxes)
        
        # Apply weights to losses
        weighted_cls_loss = (cls_loss * weights.unsqueeze(-1)).mean()
        weighted_reg_loss = (reg_loss * weights.unsqueeze(-1)).mean()
        
        total_loss = weighted_cls_loss + weighted_reg_loss
        
        return total_loss, weighted_cls_loss, weighted_reg_loss
    
    @staticmethod
    def calculate_iou(boxes1, boxes2):
        """
        Calculate IoU between two sets of boxes
        boxes: [N, 4] format (x1, y1, x2, y2)
        """
        # Calculate intersection
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou


class YOLO_EMA_SlideLoss(nn.Module):
    """
    EMA-SlideLoss adapted for YOLO format
    Works with YOLO's loss computation
    """
    def __init__(self, num_classes=5, decay=0.9, num_iterations=1000):
        super().__init__()
        self.num_classes = num_classes
        self.decay = decay
        self.num_iterations = num_iterations
        self.register_buffer('ema_mu', torch.tensor(0.5))
        self.register_buffer('iteration', torch.tensor(0))
    
    def update_ema_mu(self, current_mu):
        """Update EMA of mu"""
        self.iteration += 1
        t = self.iteration.float()
        t_all = self.num_iterations
        
        decay_factor = self.decay * (1 - torch.exp(-t / t_all))
        self.ema_mu = decay_factor * self.ema_mu + (1 - self.decay) * current_mu
    
    def slide_weight(self, iou):
        """Calculate slide weight"""
        ema_mu = self.ema_mu.item()
        weights = torch.ones_like(iou)
        
        mask1 = iou <= (ema_mu - 0.1)
        weights[mask1] = 1.0
        
        mask2 = (iou > (ema_mu - 0.1)) & (iou < ema_mu)
        weights[mask2] = torch.exp(torch.tensor(1.0 - ema_mu))
        
        mask3 = iou >= ema_mu
        weights[mask3] = torch.exp(1.0 - iou[mask3])
        
        return weights
    
    def forward(self, predictions, targets, ious=None):
        """
        Calculate loss with EMA-SlideLoss weighting
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            ious: Pre-computed IoUs (optional)
        
        Returns:
            Weighted loss
        """
        # If IoUs not provided, compute from predictions and targets
        if ious is None:
            # Extract boxes and compute IoU
            # This is a simplified version - in practice, need proper box extraction
            ious = torch.ones(predictions.shape[0], device=predictions.device) * 0.5
        
        # Update EMA mu
        current_mu = ious.mean().item()
        self.update_ema_mu(torch.tensor(current_mu))
        
        # Calculate weights
        weights = self.slide_weight(ious)
        
        return weights

