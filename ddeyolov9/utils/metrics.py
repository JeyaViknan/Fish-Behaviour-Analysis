"""
Evaluation metrics for object detection
"""
import torch
import numpy as np
from collections import defaultdict


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    Box format: [x_center, y_center, width, height] (normalized)
    """
    # Convert to [x1, y1, x2, y2]
    def to_corners(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = to_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = to_corners(box2)
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def calculate_map(predictions, targets, num_classes=5, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of predictions, each is dict with 'boxes', 'scores', 'labels'
        targets: List of targets, each is dict with 'boxes', 'labels'
        num_classes: Number of classes
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        mAP value
    """
    aps = []
    
    for class_id in range(num_classes):
        # Collect all predictions and targets for this class
        pred_boxes = []
        pred_scores = []
        target_boxes = []
        
        for pred, target in zip(predictions, targets):
            # Filter by class
            pred_mask = pred['labels'] == class_id
            target_mask = target['labels'] == class_id
            
            if pred_mask.any():
                pred_boxes.extend(pred['boxes'][pred_mask].cpu().numpy())
                pred_scores.extend(pred['scores'][pred_mask].cpu().numpy())
            
            if target_mask.any():
                target_boxes.extend(target['boxes'][target_mask].cpu().numpy())
        
        if len(pred_boxes) == 0:
            aps.append(0.0)
            continue
        
        # Sort predictions by score
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = np.array(pred_boxes)[sorted_indices]
        pred_scores = np.array(pred_scores)[sorted_indices]
        target_boxes = np.array(target_boxes)
        
        # Calculate TP and FP
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        target_matched = np.zeros(len(target_boxes), dtype=bool)
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_target_idx = -1
            
            for j, target_box in enumerate(target_boxes):
                if target_matched[j]:
                    continue
                
                iou = calculate_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                target_matched[best_target_idx] = True
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / (len(target_boxes) + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        aps.append(ap)
    
    return np.mean(aps), aps


def calculate_precision_recall(predictions, targets, num_classes=5, iou_threshold=0.5):
    """
    Calculate precision and recall
    
    Returns:
        precision, recall (per class and overall)
    """
    tp_total = defaultdict(int)
    fp_total = defaultdict(int)
    fn_total = defaultdict(int)
    
    for pred, target in zip(predictions, targets):
        for class_id in range(num_classes):
            pred_mask = pred['labels'] == class_id
            target_mask = target['labels'] == class_id
            
            pred_boxes = pred['boxes'][pred_mask] if pred_mask.any() else torch.tensor([])
            target_boxes = target['boxes'][target_mask] if target_mask.any() else torch.tensor([])
            
            if len(pred_boxes) == 0 and len(target_boxes) == 0:
                continue
            
            # Match predictions to targets
            if len(pred_boxes) > 0 and len(target_boxes) > 0:
                matched_targets = set()
                for pred_box in pred_boxes:
                    best_iou = 0.0
                    best_target_idx = -1
                    
                    for j, target_box in enumerate(target_boxes):
                        if j in matched_targets:
                            continue
                        iou = calculate_iou(pred_box.cpu().numpy(), target_box.cpu().numpy())
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = j
                    
                    if best_iou >= iou_threshold:
                        tp_total[class_id] += 1
                        matched_targets.add(best_target_idx)
                    else:
                        fp_total[class_id] += 1
                fn_total[class_id] += len(target_boxes) - len(matched_targets)
            elif len(pred_boxes) > 0:
                fp_total[class_id] += len(pred_boxes)
            elif len(target_boxes) > 0:
                fn_total[class_id] += len(target_boxes)
    
    # Calculate precision and recall per class
    precisions = {}
    recalls = {}
    
    for class_id in range(num_classes):
        tp = tp_total[class_id]
        fp = fp_total[class_id]
        fn = fn_total[class_id]
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        precisions[class_id] = precision
        recalls[class_id] = recall
    
    # Overall precision and recall
    total_tp = sum(tp_total.values())
    total_fp = sum(fp_total.values())
    total_fn = sum(fn_total.values())
    
    overall_precision = total_tp / (total_tp + total_fp + 1e-6)
    overall_recall = total_tp / (total_tp + total_fn + 1e-6)
    
    return overall_precision, overall_recall, precisions, recalls

