"""
Flask backend for fish behavior detection.
Runs YOLO inference and returns detections with behavior meanings.
"""
import os
import base64
import io
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Behavior definitions: class_id -> { label, meaning, is_anomaly, color }
BEHAVIOR_MEANINGS = {
    0: {
        "label": "Normal-Fish",
        "meaning": "Normal swimming behavior; fish is healthy.",
        "is_anomaly": False,
        "color": "#22c55e",  # green
    },
    1: {
        "label": "PH abnormal-Fish",
        "meaning": "Convulsion: abnormal pH (e.g. weakly acidic water) may cause twitching or convulsive movements.",
        "is_anomaly": True,
        "color": "#eab308",  # yellow
    },
    2: {
        "label": "Low temperature-Fish",
        "meaning": "Head down, tail up: cold stress (below ~15°C) can cause this posture.",
        "is_anomaly": True,
        "color": "#3b82f6",  # blue
    },
    3: {
        "label": "High temperature-Fish",
        "meaning": "Rollover: heat stress (above ~25°C) can lead to loss of balance or rollover.",
        "is_anomaly": True,
        "color": "#ef4444",  # red
    },
    4: {
        "label": "Hypoxia-Fish",
        "meaning": "Head up, tail down: low dissolved oxygen; fish may tilt to gulp air at surface.",
        "is_anomaly": True,
        "color": "#a855f7",  # purple
    },
}

app = Flask(__name__, static_folder="static")
CORS(app)

# Load YOLO model once
MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    # Prefer trained weights, fallback to pretrained
    candidates = [
        Path(__file__).resolve().parent.parent / "runs" / "train" / "ddeyolov9" / "weights" / "best.pt",
        Path(__file__).resolve().parent.parent / "weights" / "best.pt",
        "yolov9e.pt",
    ]
    for p in candidates:
        if p and Path(p).exists():
            MODEL_PATH = str(p)
            break
    else:
        MODEL_PATH = "yolov9e.pt"

model = None


def get_model():
    global model
    if model is None:
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")
    return model


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files and not request.json:
        return jsonify({"error": "No image provided"}), 400

    try:
        if request.files.get("image"):
            file = request.files["image"]
            img_bytes = file.read()
        elif request.json and "image" in request.json:
            # base64 from frontend
            img_bytes = base64.b64decode(request.json["image"].split(",")[-1])
        else:
            return jsonify({"error": "No image in request"}), 400

        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        yolo = get_model()
        results = yolo(img_np, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                info = BEHAVIOR_MEANINGS.get(cls, {
                    "label": f"Class-{cls}",
                    "meaning": "Unknown behavior.",
                    "is_anomaly": False,
                    "color": "#6b7280",
                })
                detections.append({
                    "bbox": xyxy,
                    "confidence": round(conf, 4),
                    "class_id": cls,
                    "label": info["label"],
                    "meaning": info["meaning"],
                    "is_anomaly": info["is_anomaly"],
                    "color": info["color"],
                })

        return jsonify({
            "detections": detections,
            "behavior_meanings": BEHAVIOR_MEANINGS,
            "anomalies_detected": any(d["is_anomaly"] for d in detections),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/meanings", methods=["GET"])
def meanings():
    return jsonify({"behavior_meanings": BEHAVIOR_MEANINGS})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    app.run(host="0.0.0.0", port=args.port, debug=True)
