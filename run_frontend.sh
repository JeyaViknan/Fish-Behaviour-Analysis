#!/bin/bash
# Run the fish behavior detection frontend
cd "$(dirname "$0")"

# Prefer trained model if available
if [ -f "runs/train/ddeyolov9/weights/best.pt" ]; then
  export MODEL_PATH="$(pwd)/runs/train/ddeyolov9/weights/best.pt"
  echo "Using trained model: $MODEL_PATH"
elif [ -f "weights/best.pt" ]; then
  export MODEL_PATH="$(pwd)/weights/best.pt"
  echo "Using trained model: $MODEL_PATH"
else
  echo "Using default YOLOv9 model (yolov9e.pt). Train first for best results."
fi

cd app
# Use 5050 if 5000 is in use (e.g. macOS AirPlay)
python3 backend.py --port "${PORT:-5050}"
