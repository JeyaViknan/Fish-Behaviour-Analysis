# Streamlit App Guide - Pufferfish Behavior Detection

## 🚀 Running the App

The Streamlit app is now running at: **http://localhost:8501**

To run it again:
```bash
./run_streamlit.sh
# or
streamlit run streamlit_app.py
```

## 📋 Features

### 1. 📊 Dataset Explorer
- **Statistics Dashboard**: View total images, labels, objects, and averages
- **Class Distribution**: Interactive bar chart showing distribution of all 5 behavior classes
- **Behavior Classifications Table**: Detailed breakdown of each class with counts and percentages
- **Image Browser**: Browse through dataset images with annotations overlaid
- **Visual Annotations**: See bounding boxes and labels directly on images

### 2. 🎯 Test Model - Anomaly Detection Only
- **Upload Images**: Upload fish images for detection
- **Dataset Samples**: Select from your dataset for quick testing
- **Anomaly-Only Display**: 
  - ✅ **Shows results ONLY when anomalies are detected**
  - ✅ Normal fish detections are hidden unless no anomalies found
  - ✅ Clear anomaly warnings and detailed explanations
- **Anomaly Details**: Expandable cards showing:
  - Condition (e.g., "pH abnormal", "Low temperature")
  - Description of the behavior
  - Meaning and implications
- **Summary Table**: Overview of all detected anomalies

### 3. 🚀 Train Model
- **Interactive Training**: Configure training parameters:
  - Epochs (10-300, default: 200)
  - Batch size (4, 8, 16)
  - Image size (640, 512, 416)
  - Learning rate
  - Train/Val split ratio
- **Real-time Training**: Start training directly from the app
- **Results Display**: View training metrics (mAP, Precision, Recall)

## 🐡 Behavior Classifications

| Class | Name | Type | Condition | Color |
|-------|------|------|-----------|-------|
| 0 | Normal-Fish | Normal | Normal water conditions | 🟢 Green |
| 1 | PH abnormal-Fish | **Anomaly** | pH abnormal (weakly acidic) | 🟡 Yellow |
| 2 | Low temperature-Fish | **Anomaly** | Low temperature (<15°C) | 🔵 Blue |
| 3 | High temperature-Fish | **Anomaly** | High temperature (>25°C) | 🔴 Red |
| 4 | Hypoxia-Fish | **Anomaly** | Hypoxia (low dissolved oxygen) | 🟣 Purple |

## 🎯 Key Features for Anomaly Detection

### Anomaly-Only Results
- The test page **only displays results when anomalies are detected**
- Normal fish are filtered out from visualization
- Clear warnings and alerts for detected anomalies
- Detailed explanations of what each anomaly means

### Behavior Meanings
When an anomaly is detected, the app shows:
1. **Visual Detection**: Bounding boxes with colored labels
2. **Condition**: What environmental condition caused it
3. **Description**: What the behavior looks like
4. **Meaning**: Why this matters and what action to take

## 📊 Dataset Information

The app automatically loads:
- All images from `images/` directory
- All labels from `labels/` directory
- Class distribution statistics
- Per-image annotations

## 🔧 Model Loading

The app tries to load models in this order:
1. `runs/train/ddeyolov9/weights/best.pt` (trained model)
2. `weights/best.pt` (alternative trained model)
3. `yolov9e.pt` (pretrained fallback)

## 💡 Usage Tips

1. **First Time**: Train the model using the "Train Model" page for best results
2. **Testing**: Use the "Test Model" page to upload images
3. **Dataset Exploration**: Browse your dataset to understand class distribution
4. **Anomaly Focus**: The test page automatically filters to show only anomalies

## 🎨 Visual Features

- **Color-coded detections**: Each behavior type has a unique color
- **Interactive charts**: Plotly charts for data visualization
- **Image annotations**: See bounding boxes directly on images
- **Responsive layout**: Works on different screen sizes

---

**The app is running at: http://localhost:8501**

Open it in your browser to start exploring!
