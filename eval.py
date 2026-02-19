"""
Evaluation script for DDEYOLOv9
"""
import torch
from ultralytics import YOLO
from ddeyolov9.utils.metrics import calculate_map, calculate_precision_recall
from ddeyolov9.data.dataset import FishBehaviorDataset
from torch.utils.data import DataLoader
import os


def evaluate_model(model_path, images_dir, labels_dir, img_size=640, batch_size=8):
    """
    Evaluate trained model
    
    Args:
        model_path: Path to trained model weights
        images_dir: Directory containing test images
        labels_dir: Directory containing test labels
        img_size: Image size
        batch_size: Batch size for evaluation
    """
    # Load model
    model = YOLO(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create test dataset
    test_dataset = FishBehaviorDataset(images_dir, labels_dir, img_size=img_size, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    model.eval()
    predictions = []
    targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['images'].to(device)
            boxes = batch['boxes']
            labels = batch['labels']
            
            # Get predictions
            results = model(images)
            
            # Process results (adapt based on YOLO output format)
            for i, result in enumerate(results):
                # Extract predictions from YOLO result
                # This needs to be adapted based on actual YOLO output
                pass
    
    # Calculate metrics
    map_value, map_per_class = calculate_map(predictions, targets, num_classes=5)
    precision, recall, precisions, recalls = calculate_precision_recall(predictions, targets, num_classes=5)
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Overall mAP: {map_value:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print("\nPer-class mAP:")
    class_names = ['Normal-Fish', 'PH abnormal-Fish', 'Low temperature-Fish', 
                   'High temperature-Fish', 'Hypoxia-Fish']
    for i, (name, map_val) in enumerate(zip(class_names, map_per_class)):
        print(f"  {name}: {map_val:.4f}")
    print("\nPer-class Precision:")
    for i, (name, prec) in enumerate(zip(class_names, precisions.values())):
        print(f"  {name}: {prec:.4f}")
    print("\nPer-class Recall:")
    for i, (name, rec) in enumerate(zip(class_names, recalls.values())):
        print(f"  {name}: {rec:.4f}")
    print("="*50)
    
    return {
        'map': map_value,
        'map_per_class': map_per_class,
        'precision': precision,
        'recall': recall,
        'precisions': precisions,
        'recalls': recalls
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DDEYOLOv9 model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--images', type=str, default='images', help='Images directory')
    parser.add_argument('--labels', type=str, default='labels', help='Labels directory')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        images_dir=args.images,
        labels_dir=args.labels,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

