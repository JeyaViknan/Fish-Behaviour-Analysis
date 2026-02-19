"""
Dataset loader for fish abnormal behavior detection
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FishBehaviorDataset(Dataset):
    """
    Dataset for fish abnormal behavior detection
    Supports YOLO format annotations
    """
    def __init__(self, images_dir, labels_dir, img_size=640, augment=False, split='train'):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        self.split = split
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        # Filter out images without corresponding labels
        self.image_files = [f for f in self.image_files 
                           if os.path.exists(os.path.join(labels_dir, f.replace('.jpg', '.txt')))]
        
        # Data augmentation
        if self.augment and split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomSizedCrop(min_max_height=(int(img_size*0.8), img_size), height=img_size, width=img_size, p=0.3),
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # Load labels
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        boxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.int64) if class_labels else np.zeros((0,), dtype=np.int64)
        
        # Apply augmentations
        if len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes']) if transformed['bboxes'] else np.zeros((0, 4), dtype=np.float32)
            class_labels = np.array(transformed['class_labels']) if transformed['class_labels'] else np.zeros((0,), dtype=np.int64)
        else:
            # No boxes, just transform image
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros((0,), dtype=np.int64)
        
        # Convert boxes to tensor format [x_center, y_center, width, height] (normalized)
        boxes_tensor = torch.from_numpy(boxes).float() if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.from_numpy(class_labels).long() if len(class_labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        
        return {
            'image': image,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'img_path': img_path
        }


def create_dataloaders(images_dir, labels_dir, img_size=640, batch_size=8, num_workers=4, train_ratio=0.8, val_ratio=0.1):
    """
    Create train, validation, and test dataloaders
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        img_size: Image size for resizing
        batch_size: Batch size
        num_workers: Number of worker processes
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    import random
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    image_files = [f for f in image_files 
                   if os.path.exists(os.path.join(labels_dir, f.replace('.jpg', '.txt')))]
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    
    total_size = len(image_files)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # Create datasets
    train_dataset = FishBehaviorDataset(images_dir, labels_dir, img_size=img_size, augment=True, split='train')
    train_dataset.image_files = train_files
    
    val_dataset = FishBehaviorDataset(images_dir, labels_dir, img_size=img_size, augment=False, split='val')
    val_dataset.image_files = val_files
    
    test_dataset = FishBehaviorDataset(images_dir, labels_dir, img_size=img_size, augment=False, split='test')
    test_dataset.image_files = test_files
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'img_paths': img_paths
    }

