# Quick Start Guide for DDEYOLOv9

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Quick Training (Simplified - Using Ultralytics YOLOv9)

The easiest way to get started is using the simplified training script:

```bash
python train_yolo.py
```

This will:
1. Automatically create `data.yaml` configuration
2. Download YOLOv9-E pretrained weights (if not present)
3. Train on your dataset
4. Validate and export the model

## Full Implementation Steps

### 1. Verify Dataset Structure

Your dataset should be organized as:
```
.
├── images/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── labels/
    ├── 0.txt
    ├── 1.txt
    └── ...
```

### 2. Check Data Configuration

The `data.yaml` file should have:
- 5 classes (Normal, PH abnormal, Low temperature, High temperature, Hypoxia)
- Correct paths to images and labels

### 3. Training Options

#### Option A: Quick Start (Recommended)
```bash
python train_yolo.py
```

#### Option B: Full Custom Training
```bash
python train_ddeyolov9.py
```

### 4. Evaluation

After training, evaluate your model:
```bash
python eval.py --model runs/train/ddeyolov9/weights/best.pt --images images --labels labels
```

## Expected Training Time

- **400 images**: ~2-4 hours on GPU (RTX 3070 Ti or similar)
- **200 epochs**: Full training cycle
- **Batch size 8**: Adjust based on GPU memory

## Monitoring Training

Training logs and results will be saved in:
- `runs/train/ddeyolov9/` - Training outputs
- `weights/` - Model checkpoints (if using custom training)

## Key Parameters

Based on the paper:
- **Learning Rate**: 0.01 (initial)
- **Batch Size**: 8
- **Image Size**: 640×640
- **Epochs**: 200
- **Optimizer**: SGD with momentum 0.937

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `batch=4` or `batch=2`
- Reduce image size: `imgsz=512`

### Low Accuracy
- Your dataset has 400 images (vs 4000 in paper)
- Consider:
  - More aggressive data augmentation
  - Transfer learning from pretrained models
  - Collecting more data

### Class Imbalance
- Class 0 (Normal) dominates with 95.2%
- EMA-SlideLoss helps, but you may also:
  - Use class weights
  - Oversample rare classes
  - Use focal loss

## Next Steps

1. **Train the model**: Run `python train_yolo.py`
2. **Evaluate results**: Check metrics in training output
3. **Fine-tune**: Adjust hyperparameters if needed
4. **Deploy**: Export model for inference

## Integration Notes

To fully integrate all three improvements (DRNELAN4, DCNv4-Dyhead, EMA-SlideLoss):

1. **Modify YOLOv9 Architecture**:
   - Replace `RepNCSPELAN4` with `DRNELAN4` in backbone
   - Replace detection head with `DCNv4Dyhead`

2. **Integrate Loss Function**:
   - Modify YOLO's loss calculation to use `EMA_SlideLoss`
   - Or use as a wrapper around existing loss

3. **Custom Implementation**:
   - Use the provided modules in a custom YOLO implementation
   - Integrate with other object detection frameworks

See the README.md for more details on each component.

