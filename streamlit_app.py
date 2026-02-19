"""
Streamlit Frontend for Fish Behavior Detection
Shows dataset, classifications, and testing with anomaly-only results
"""
import streamlit as st
import os
import cv2
import numpy as np
import re
from pathlib import Path
import pandas as pd
from PIL import Image
import plotly.express as px
from ultralytics import YOLO

from streamlit_utils import (
    build_train_val_split,
    compute_class_counts,
    parse_label_line,
    write_data_yaml,
    write_split_files,
)

# Page config
st.set_page_config(
    page_title="Pufferfish Behavior Detection",
    page_icon="🐡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Behavior definitions
BEHAVIOR_INFO = {
    0: {
        "name": "Normal-Fish",
        "description": "Normal swimming behavior; fish is healthy.",
        "is_anomaly": False,
        "color": "#22c55e",
        "condition": "Normal water conditions"
    },
    1: {
        "name": "PH abnormal-Fish",
        "description": "Convulsion: Abnormal pH (weakly acidic water) causes twitching or convulsive movements.",
        "is_anomaly": True,
        "color": "#eab308",
        "condition": "pH abnormal (weakly acidic)"
    },
    2: {
        "name": "Low temperature-Fish",
        "description": "Head down, tail up: Cold stress below ~15°C causes this posture.",
        "is_anomaly": True,
        "color": "#3b82f6",
        "condition": "Low temperature (<15°C)"
    },
    3: {
        "name": "High temperature-Fish",
        "description": "Rollover: Heat stress above ~25°C leads to loss of balance or rollover.",
        "is_anomaly": True,
        "color": "#ef4444",
        "condition": "High temperature (>25°C)"
    },
    4: {
        "name": "Hypoxia-Fish",
        "description": "Head up, tail down: Low dissolved oxygen; fish tilts to gulp air at surface.",
        "is_anomaly": True,
        "color": "#a855f7",
        "condition": "Hypoxia (low dissolved oxygen)"
    }
}

# Paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
MODEL_DIR = BASE_DIR / "runs" / "train" / "ddeyolov9" / "weights"
WEIGHTS_DIR = BASE_DIR / "weights"
FALLBACK_MODEL = BASE_DIR / "yolov9e.pt"


def get_runtime_compatibility_issue():
    """Return a human-readable compatibility issue, or None when runtime is OK."""
    try:
        import torch
    except Exception:
        return None

    try:
        np_major = int(str(np.__version__).split(".")[0])
    except Exception:
        np_major = None

    torch_version = str(torch.__version__)
    match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if np_major is None or not match:
        return None

    torch_major = int(match.group(1))
    torch_minor = int(match.group(2))

    # Torch <= 2.1 wheels are commonly built against NumPy 1.x and fail with NumPy 2.x.
    if np_major >= 2 and (torch_major < 2 or (torch_major == 2 and torch_minor <= 1)):
        return (
            f"Incompatible runtime detected: torch {torch_version} with numpy {np.__version__}. "
            "Install `numpy<2` for this environment (or upgrade torch to a build compatible with NumPy 2)."
        )

    return None

@st.cache_resource
def load_model():
    """Load YOLO model"""
    model_paths = [
        MODEL_DIR / "best.pt",
        WEIGHTS_DIR / "best.pt",
        FALLBACK_MODEL,
    ]
    
    for path in model_paths:
        if not path.exists():
            continue
        try:
            model = YOLO(str(path))
            is_trained = path.name == "best.pt"
            return model, is_trained
        except Exception:
            continue
    
    st.error("❌ Could not load model")
    return None, False

def load_dataset_info():
    """Load dataset statistics"""
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        return None
    
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])
    labels = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')])
    
    label_paths = [LABELS_DIR / label_file for label_file in labels]
    class_counts, total_objects = compute_class_counts(label_paths, num_classes=5)
    
    return {
        "total_images": len(images),
        "total_labels": len(labels),
        "total_objects": total_objects,
        "class_counts": class_counts,
        "images": images[:100]  # Limit for display
    }

def draw_boxes(image, detections, show_all=False):
    """Draw bounding boxes on image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    for det in detections:
        if not show_all and not det['is_anomaly']:
            continue  # Skip normal fish if show_all=False
        
        x1, y1, x2, y2 = det['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = tuple(int(det['color'][i:i+2], 16) for i in (1, 3, 5))
        color = (color[2], color[1], color[0])  # BGR for OpenCV
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det['label']} {det['confidence']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def main():
    st.title("🐡 Pufferfish Behavior Detection System")
    st.markdown("**Detect and classify abnormal behaviors in Takifugu rubripes**")
    compat_issue = get_runtime_compatibility_issue()
    inference_only_mode = compat_issue is not None

    if inference_only_mode:
        st.warning(
            "Inference-only mode is active. Training is temporarily disabled due to a runtime dependency mismatch."
        )
        st.caption(f"Details: {compat_issue}")
        st.code("python3.11 -m pip install --user 'numpy<2'")
    
    # Sidebar
    st.sidebar.title("Navigation")
    if inference_only_mode:
        st.sidebar.warning("Training disabled in inference-only mode.")
        page_options = ["📊 Dataset Explorer", "🎯 Test Model"]
    else:
        page_options = ["📊 Dataset Explorer", "🎯 Test Model", "🚀 Train Model"]

    page = st.sidebar.radio(
        "Select Page",
        page_options
    )
    
    if page == "📊 Dataset Explorer":
        show_dataset_explorer()
    elif page == "🎯 Test Model":
        show_test_model()
    elif page == "🚀 Train Model":
        show_train_model()

def show_dataset_explorer():
    st.header("📊 Dataset Explorer")
    
    dataset_info = load_dataset_info()
    if not dataset_info:
        st.error("Dataset not found. Please ensure 'images' and 'labels' directories exist.")
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", dataset_info["total_images"])
    with col2:
        st.metric("Total Labels", dataset_info["total_labels"])
    with col3:
        st.metric("Total Objects", dataset_info["total_objects"])
    with col4:
        st.metric("Avg Objects/Image", f"{dataset_info['total_objects']/dataset_info['total_images']:.1f}")
    
    # Class distribution
    st.subheader("Class Distribution")
    class_data = []
    for class_id, count in dataset_info["class_counts"].items():
        info = BEHAVIOR_INFO[class_id]
        class_data.append({
            "Class": info["name"],
            "Count": count,
            "Percentage": (count / dataset_info["total_objects"] * 100) if dataset_info["total_objects"] > 0 else 0,
            "Type": "Anomaly" if info["is_anomaly"] else "Normal",
            "Color": info["color"]
        })
    
    df = pd.DataFrame(class_data)
    
    # Bar chart
    fig = px.bar(
        df,
        x="Class",
        y="Count",
        color="Type",
        color_discrete_map={"Normal": "#22c55e", "Anomaly": "#ef4444"},
        title="Distribution of Fish Behaviors"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Class details table
    st.subheader("Behavior Classifications")
    st.dataframe(df[["Class", "Count", "Percentage", "Type"]], use_container_width=True)
    
    # Image viewer
    st.subheader("Browse Dataset")
    selected_image = st.selectbox("Select an image", dataset_info["images"])
    
    if selected_image:
        img_path = IMAGES_DIR / selected_image
        label_path = LABELS_DIR / selected_image.replace('.jpg', '.txt')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Image**")
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
        
        with col2:
            st.write("**With Annotations**")
            if label_path.exists():
                img_annotated = img_rgb.copy()
                with open(label_path, 'r') as f:
                    for line in f:
                        parsed = parse_label_line(line)
                        if parsed is None:
                            continue
                        class_id, (x_center, y_center, width, height) = parsed
                        info = BEHAVIOR_INFO.get(class_id)
                        if info is None:
                            continue
                        
                        h, w = img_annotated.shape[:2]
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        color = tuple(int(info["color"][i:i+2], 16) for i in (1, 3, 5))
                        color = (color[2], color[1], color[0])
                        
                        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
                        label = info["name"]
                        cv2.putText(img_annotated, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                st.image(img_annotated, use_container_width=True)
                
                # Show labels
                st.write("**Detections:**")
                with open(label_path, 'r') as f:
                    for i, line in enumerate(f):
                        parsed = parse_label_line(line)
                        if parsed is None:
                            continue
                        class_id, _ = parsed
                        info = BEHAVIOR_INFO.get(class_id)
                        if info is None:
                            continue
                        st.write(f"- **{info['name']}** ({'Anomaly' if info['is_anomaly'] else 'Normal'})")
            else:
                st.warning("No labels found for this image")

def show_test_model():
    st.header("🎯 Test Model - Anomaly Detection")
    st.markdown("**Upload an image to detect abnormal fish behaviors. Results shown only when anomalies are detected.**")
    
    # Load model
    model, is_trained = load_model()
    if model is None:
        st.error("Model not available. Please train the model first.")
        return

    compat_issue = get_runtime_compatibility_issue()
    if compat_issue:
        st.error(f"Runtime dependency issue: {compat_issue}")
        st.code("python3.11 -m pip install --user 'numpy<2'")
        st.info("Restart Streamlit after installing compatible versions.")
        return
    
    if is_trained:
        st.success("✅ Loaded trained model from `runs/` or `weights/`.")
    else:
        st.warning("⚠️ Using pretrained model. For best results, train on your dataset first.")

    col_left, col_right = st.columns(2)
    with col_left:
        conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    with col_right:
        show_normal_if_no_anomaly = st.checkbox(
            "Show normal detections when no anomaly is found",
            value=True,
        )
    
    # Upload section
    uploaded_file = st.file_uploader("Upload fish image", type=['jpg', 'jpeg', 'png'])
    
    # Or use sample from dataset
    dataset_info = load_dataset_info()
    if dataset_info and dataset_info["images"]:
        st.write("**OR select from dataset:**")
        sample_image = st.selectbox("Select sample image", [None] + dataset_info["images"])
    else:
        sample_image = None
    
    if uploaded_file or sample_image:
        # Load image
        if sample_image:
            image = Image.open(IMAGES_DIR / sample_image)
        else:
            image = Image.open(uploaded_file)
        image = image.convert("RGB")
        img_array = np.array(image)
        
        # Run detection
        with st.spinner("Analyzing image..."):
            results = model(img_array, conf=conf_threshold, verbose=False)
        
        # Process results
        detections = []
        for r in results:
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    box = r.boxes[i]
                    class_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()
                    
                    info = BEHAVIOR_INFO.get(class_id)
                    if info is None:
                        continue
                    detections.append({
                        "bbox": xyxy,
                        "confidence": conf,
                        "class_id": class_id,
                        "label": info["name"],
                        "description": info["description"],
                        "condition": info["condition"],
                        "is_anomaly": info["is_anomaly"],
                        "color": info["color"]
                    })
        
        # Check if anomalies detected
        anomalies = [d for d in detections if d["is_anomaly"]]
        normal = [d for d in detections if not d["is_anomaly"]]
        
        if anomalies:
            st.success(f"⚠️ **{len(anomalies)} Anomalous Behavior(s) Detected!**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Result")
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_annotated = draw_boxes(img_cv, detections, show_all=False)
                img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
                st.image(img_annotated_rgb, use_container_width=True)
            
            with col2:
                st.subheader("Anomaly Details")
                for anomaly in anomalies:
                    with st.expander(f"🔴 {anomaly['label']} (Confidence: {anomaly['confidence']:.2%})", expanded=True):
                        st.markdown(f"**Condition:** {anomaly['condition']}")
                        st.markdown(f"**Description:** {anomaly['description']}")
                        st.markdown(f"**Meaning:** This behavior indicates an abnormal environmental condition that requires attention.")
            
            # Summary
            st.subheader("📋 Summary")
            summary_data = []
            for anomaly in anomalies:
                summary_data.append({
                    "Behavior": anomaly['label'],
                    "Condition": anomaly['condition'],
                    "Confidence": f"{anomaly['confidence']:.2%}"
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
        else:
            if detections:
                st.info(f"✅ **{len(normal)} Normal Fish Detected** - No anomalies found.")
                if show_normal_if_no_anomaly:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    img_annotated = draw_boxes(img_cv, detections, show_all=True)
                    img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
                    st.image(img_annotated_rgb, use_container_width=True)
            else:
                st.warning("No fish detected in the image.")

def show_train_model():
    st.header("🚀 Train Model")
    st.markdown("Train DDEYOLOv9 on your dataset for optimal anomaly detection.")
    
    dataset_info = load_dataset_info()
    if not dataset_info:
        st.error("Dataset not found. Please ensure 'images' and 'labels' directories exist.")
        return
    
    st.info(f"📊 Dataset ready: {dataset_info['total_images']} images, {dataset_info['total_objects']} annotations")

    compat_issue = get_runtime_compatibility_issue()
    if compat_issue:
        st.error(f"Runtime dependency issue: {compat_issue}")
        st.code("python3.11 -m pip install --user 'numpy<2'")
        st.info("Training requires fixing this dependency mismatch and restarting Streamlit.")
        return
    
    # Training parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", 10, 300, 200)
        batch_size = st.selectbox("Batch Size", [4, 8, 16], index=1)
        img_size = st.selectbox("Image Size", [640, 512, 416], index=0)
    with col2:
        learning_rate = st.number_input("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        train_ratio = st.slider("Train/Val Split", 0.7, 0.9, 0.8, 0.05)
    
    random_seed = st.number_input("Split Random Seed", 0, 9999, 42, 1)

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model... This may take several hours."):
            try:
                model = YOLO("yolov9e.pt")
                
                image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
                train_images, val_images = build_train_val_split(
                    image_paths=image_paths,
                    train_ratio=train_ratio,
                    seed=int(random_seed),
                )
                train_txt, val_txt = write_split_files(BASE_DIR, train_images, val_images)

                # Always refresh data.yaml to match split inputs.
                data_yaml = BASE_DIR / "data.yaml"
                class_names = [BEHAVIOR_INFO[i]["name"] for i in range(5)]
                write_data_yaml(
                    target_path=data_yaml,
                    base_dir=BASE_DIR,
                    train_ref=train_txt,
                    val_ref=val_txt,
                    class_names=class_names,
                )

                st.caption(
                    f"Using split with {len(train_images)} train images and {len(val_images)} val images."
                )
                
                results = model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    lr0=learning_rate,
                    project=str(BASE_DIR / "runs" / "train"),
                    name="ddeyolov9",
                    exist_ok=True
                )
                
                st.success("✅ Training completed!")
                st.json({
                    "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    "Precision": float(results.results_dict.get('metrics/precision(B)', 0)),
                    "Recall": float(results.results_dict.get('metrics/recall(B)', 0))
                })
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()
