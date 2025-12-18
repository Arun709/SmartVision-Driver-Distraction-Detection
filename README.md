# 🚗 SmartVision: Real-Time Driver Distraction Detection System

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://arunsmartvisionai.streamlit.app/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)](https://www.python.org/)

> **Executive-Ready AI Intelligence for Driver Safety**

SmartVision is an advanced computer vision system that detects distracted driving in real-time using YOLOv8-based object detection with custom logic-based inference. It goes beyond simple classification by analyzing physical object intersections (e.g., detecting when a phone overlaps with a hand) to reduce false positives and deliver production-grade accuracy.

---

## 🎯 Problem Statement

Driver distraction is a leading cause of accidents. Traditional object detection models struggle with:
- **High false positives**: Flagging phones present but not in use
- **Limited context**: Can't distinguish between phone presence and actual phone usage
- **Real-time performance**: Difficulty processing live video streams efficiently

SmartVision solves this with **logic-based post-processing on YOLO outputs**, achieving reliable, low-latency detection suitable for fleet monitoring and safety compliance.

---

## 🌟 Key Features

### 1. 🧠 Logic-Based Detection Engine
- **Intersection Analysis**: A phone is flagged as a threat only if it physically overlaps with a hand bounding box (calculated using Intersection over Union - IoU)
- **Reduced False Positives**: Drastically improves reliability over standard YOLO classification
- **Business-Ready**: Demonstrates custom post-processing and business logic integration

### 2. 🎨 "Onyx & Neon" Executive-Grade UI
- **Glass-Morphism Design**: Dark, futuristic theme with modern aesthetics
- **Real-Time Dashboard**: Live video feed with threat indicators and analytics
- **Presentation-Ready**: Built for C-suite demos and stakeholder presentations

### 3. 🎛️ Live Sensitivity Control
- **Dynamic Confidence Slider** (0.0 to 1.0): Adjust model sensitivity on-the-fly without restarting
- **Debug-Friendly**: Perfect for tuning during live demos
- **Interactive Feedback**: See results update instantly

### 4. 📊 "Under the Hood" Explainer Mode
- **Interactive Expanders**: Educational dropdowns explaining IoU, confidence scores, neural networks
- **Non-Technical Audience**: Turns technical demos into learning experiences for CEOs, professors, and stakeholders
- **Transparency**: Clear visibility into how the AI makes decisions

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|----------|
| **Detection Model** | YOLOv8 (Nano) | SOTA object detection trained on distraction datasets |
| **UI Framework** | Streamlit | Interactive web dashboard with custom CSS |
| **Vision Library** | OpenCV (cv2) | Real-time frame processing & bounding box logic |
| **Data Processing** | Pandas & NumPy | IoU calculations, analytics, performance metrics |
| **Deployment** | Streamlit Cloud | Cloud-ready with optimized requirements.txt |
| **Language** | Python 3.10+ | End-to-end development |

---

## 📊 Model & Metrics

- **Dataset**: Custom distraction detection dataset with hand-phone interaction annotations
- **Train/Val Split**: 80/20
- **Key Metrics**:
  - **Precision**: High recall to minimize missed distraction events
  - **IoU Threshold**: 0.3 (hand-phone overlap sensitivity)
  - **Inference Speed**: ~30-40 FPS on standard GPUs

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or conda
- Webcam (for real-time demo)
- ~500MB disk space for model weights

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Arun709/SmartVision-Driver-Distraction-Detection.git
   cd SmartVision-Driver-Distraction-Detection
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights**
   
   The YOLOv8 weights are automatically downloaded on first run. If you prefer manual download:
   - Download `yolov8n.pt` from [Ultralytics Release](https://github.com/ultralytics/assets/releases)
   - Place in project root directory
   - The app will use it automatically

5. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```
   
   The app will open at: `http://localhost:8501`

### Alternative: Real-Time Demo (No Streamlit)

For a plain OpenCV demo without the web interface:
```bash
python realtime.py
```

Press `q` to quit, `s` to toggle sensitivity

---

## 📁 Project Structure

```
SmartVision-Driver-Distraction-Detection/
├── app.py                    # Streamlit web interface
├── realtime.py              # Real-time OpenCV demo
├── train_yolo.py            # YOLO model training script
├── train_classifier.py      # Phone/Hand classifier training
├── retrain_model.py         # Model retraining utilities
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── LICENSE                  # GPL-3.0 license
│
├── models/                  # Model weights directory
│   ├── yolov8n.pt          # YOLOv8 nano (auto-downloaded)
│   └── yolov8n-cls.pt      # YOLOv8 classifier
│
├── APP_UI_screenshots/      # UI screenshots for documentation
├── Yolo8_visualizations/    # Model output visualizations
└── screenshots/             # Setup and usage screenshots
```

---

## 🚀 Usage Guide

### Streamlit Web Interface

1. **Launch the App**
   ```bash
   streamlit run app.py
   ```

2. **Sidebar Controls**
   - **Confidence Threshold**: Adjust detection sensitivity (0.0-1.0)
   - **IoU Threshold**: Adjust hand-phone overlap detection (0.0-1.0)
   - **Enable Explainer**: Toggle educational mode

3. **Main Dashboard**
   - Live video feed with real-time bounding boxes
   - Threat indicator (Red = Phone + Hand overlap detected)
   - FPS and detection statistics
   - Historical analytics

### Training Your Own Model

```bash
# Train YOLOv8 on custom dataset
python train_yolo.py --epochs 50 --imgsz 640 --batch 16

# Train hand-phone classifier
python train_classifier.py --epochs 30
```

---

## 🎓 How It Works

### Detection Pipeline

1. **Frame Capture**: Real-time video from webcam/file
2. **Object Detection**: YOLOv8 detects phones and hands
3. **Logic Layer**: Calculates Intersection over Union (IoU) between bounding boxes
4. **Threat Classification**: Flags distraction only if IoU > threshold
5. **Visualization**: Renders annotated frame with alerts

### Key Algorithms

- **IoU (Intersection over Union)**:
  ```
  IoU = Intersection Area / Union Area
  If IoU > threshold → Distraction Detected
  ```

---

## 📸 Screenshots

### Live Detection Interface
![App UI Screenshot](./APP_UI_screenshots/ui_demo.png)

### Detection Visualizations
![YOLO Detection](./Yolo8_visualizations/detection_example.png)

### System Output
![System Screenshot](./screenshots/system_overview.png)

---

## 🔧 Configuration

### requirements.txt

Key dependencies:
- `streamlit>=1.28.0` - Web framework
- `yolov8>=0.0.20` - Object detection
- `opencv-python-headless>=4.8.0` - Computer vision
- `torch>=2.0.0` - Deep learning
- `pandas>=2.0.0` - Data processing

### Environment Variables

```bash
# Optional: Control camera device
CAM_INDEX=0  # Default camera

# Optional: Model paths
MODEL_PATH=./models/yolov8n.pt
CLASSIFIER_PATH=./models/yolov8n-cls.pt
```

---

## 🚢 Deployment

### Deploy to Streamlit Cloud

1. Push to GitHub (already done)
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repo
4. Select: `app.py` as main file
5. Deploy!

**Live Demo**: [arunsmartvisionai.streamlit.app](https://arunsmartvisionai.streamlit.app/)

### Deploy to Production Server

```bash
# Using Gunicorn + Nginx
gunicorn --workers 4 --worker-class sync --timeout 120 app:app
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t smartvision:latest .
docker run -p 8501:8501 smartvision:latest
```

---

## 🔍 Troubleshooting

### Issue: Model weights not downloading
**Solution**: Manually download from [Ultralytics](https://github.com/ultralytics/assets/releases) and place in project root

### Issue: Webcam not detected
**Solution**: Check camera permissions and ensure no other app is using it

### Issue: Slow performance
**Solution**: Reduce image size in code or use GPU acceleration

---

## 📈 Future Enhancements

- [ ] Multi-person detection for fleet monitoring
- [ ] Alert integration (email/SMS notifications)
- [ ] Database logging for compliance reports
- [ ] Mobile app for iOS/Android
- [ ] Edge deployment on NVIDIA Jetson
- [ ] Custom model fine-tuning via web UI
- [ ] Historical analytics dashboard

---

## 💼 Skills Demonstrated

✅ **Computer Vision**: YOLOv8, object detection, OpenCV  
✅ **Deep Learning**: PyTorch, transfer learning, model optimization  
✅ **Full-Stack Development**: Python backend, Streamlit frontend  
✅ **Data Processing**: NumPy, Pandas, real-time analytics  
✅ **UI/UX Design**: Glass-morphism, responsive design, stakeholder communication  
✅ **Business Logic**: Custom post-processing, logic-based inference  
✅ **Deployment**: Streamlit Cloud, Docker, cloud-ready architecture  

---

## 📝 License

This project is licensed under the **GPL-3.0 License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Arunachalam Kannan**
- GitHub: [@Arun709](https://github.com/Arun709)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/arunachalam-kannan)
- Email: your.email@example.com

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Submit a Pull Request

---

## ⭐ Show Your Support

If this project helped you, please:
- ⭐ Star this repository
- 🍴 Fork it
- 💬 Share feedback
- 📧 Connect on LinkedIn

---

## 📚 Resources & References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

**Last Updated**: December 18, 2025  
**Status**: ✅ Production-Ready
