# DDEYOLOv9: Network for Detecting and Counting Abnormal Fish Behaviors

Implementation of DDEYOLOv9 based on the research paper:
**"DDEYOLOv9: Network for Detecting and Counting Abnormal Fish Behaviors in Complex Water Environments"**

## Overview

DDEYOLOv9 is an enhanced YOLOv9 model with three key improvements:

1. **DRNELAN4 Module**: Replaces RepNCSPELAN4 in the backbone, using DilatedReparamBlock to improve receptive field
2. **DCNv4-Dyhead Detection Head**: Combines DCNv4 (Deformable Convolution v4) with Dynamic Head for multi-scale feature learning
3. **EMA-SlideLoss**: Handles class imbalance by focusing on hard samples using Exponential Moving Average

## Dataset

The dataset contains:
- 400 images of fish in various behaviors
- 5 classes: Normal (0), PH abnormal (1), Low temperature (2), High temperature (3), Hypoxia (4)
- YOLO format annotations
- High object density (~28 objects per image)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install ultralytics (for YOLOv9 base)
pip install ultralytics
```

## Project Structure

```
.
├── ddeyolov9/
│   ├── models/
│   │   ├── dilated_reparam.py      # DilatedReparamBlock
│   │   ├── drnelan4.py             # DRNELAN4 module
│   │   ├── dcnv4.py                 # DCNv4 implementation
│   │   ├── dcnv4_dyhead.py         # DCNv4-Dyhead detection head
│   │   ├── ema_slideloss.py         # EMA-SlideLoss loss function
│   │   └── ddeyolov9.py             # Main model
│   ├── data/
│   │   └── dataset.py               # Dataset loader
│   └── utils/
│       └── metrics.py               # Evaluation metrics
├── images/                          # Image directory
├── labels/                          # Label directory
├── train_yolo.py                    # Training script (simplified)
├── train_ddeyolov9.py               # Full training script
└── requirements.txt                 # Dependencies
```

## Usage

### Quick Start (Using Ultralytics YOLOv9)

```bash
python train_yolo.py
```

This will:
1. Create a data.yaml configuration
2. Train YOLOv9-E on your dataset
3. Validate the model
4. Export to ONNX format

### Full Implementation

To fully integrate all three improvements:

1. **Integrate DRNELAN4**: Modify YOLOv9's architecture to replace RepNCSPELAN4 with DRNELAN4
2. **Integrate DCNv4-Dyhead**: Replace the detection head with DCNv4-Dyhead
3. **Integrate EMA-SlideLoss**: Modify the loss function to use EMA-SlideLoss

See `train_ddeyolov9.py` for the training structure.

## Training Configuration

Based on the paper:
- **Model**: YOLOv9-E
- **Image Size**: 640×640
- **Batch Size**: 8
- **Learning Rate**: 0.01
- **Epochs**: 200
- **Optimizer**: SGD with momentum 0.937
- **Data Split**: 80% train, 10% val, 10% test

## Expected Results

From the paper:
- **Precision**: 91.7%
- **Recall**: 90.4%
- **mAP**: 94.1%
- **FPS**: 119

Note: With 400 images (vs 4000 in paper), results may be lower. Consider data augmentation.

## Key Components

### DRNELAN4
- Uses DilatedReparamBlock to expand receptive field
- Improves detection in complex underwater environments
- Reduces computational cost while maintaining accuracy

### DCNv4-Dyhead
- Scale-aware attention for multi-scale features
- Spatial-aware attention with DCNv4 for adaptive sampling
- Task-aware attention for different detection tasks

### EMA-SlideLoss
- Dynamically adjusts weights for easy/hard samples
- Uses Exponential Moving Average for smooth adaptation
- Handles class imbalance effectively

## Evaluation Metrics

The implementation includes:
- Precision
- Recall
- mAP (mean Average Precision)
- Per-class metrics
- FPS (Frames Per Second)

## Notes

1. **Dataset Size**: The current dataset has 400 images. The paper used 4000 images. Consider:
   - Aggressive data augmentation
   - Transfer learning from pretrained models
   - Collecting more data if possible

2. **Class Imbalance**: Class 0 (Normal) dominates with 95.2%. EMA-SlideLoss helps, but you may also consider:
   - Focal Loss
   - Class weighting
   - Oversampling rare classes

3. **Integration**: Full integration with ultralytics YOLOv9 requires modifying the framework's source code. The provided modules can be integrated into:
   - Custom YOLO implementations
   - Modified ultralytics versions
   - Standalone implementations

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{li2024ddeyolov9,
  title={DDEYOLOv9: Network for Detecting and Counting Abnormal Fish Behaviors in Complex Water Environments},
  author={Li, Yinjia and Hu, Zeyuan and Zhang, Yixi and Liu, Jihang and Tu, Wan and Yu, Hong},
  journal={Fishes},
  volume={9},
  number={6},
  pages={242},
  year={2024}
}
```

## License

This implementation is for research purposes. Please refer to the original paper and ultralytics license for usage terms.

