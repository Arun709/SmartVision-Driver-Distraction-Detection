import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
import gc
# --- NEW IMPORTS FOR WEBCAM ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av

# ==========================================
# 1. CORE CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FORCE DARK THEME & PREMIUM CSS
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at 50% 10%, #1a202c 0%, #000000 100%); color: #ffffff; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: rgba(10, 10, 10, 0.95); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    h1, h2, h3 { background: linear-gradient(to right, #00f260, #0575e6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .neo-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px; margin-bottom: 20px; }
    div.stButton > button { background: linear-gradient(135deg, #0575e6 0%, #00f260 100%); color: white; border: none; padding: 12px 24px; border-radius: 10px; font-weight: bold; width: 100%; transition: transform 0.2s; }
    div.stButton > button:hover { transform: scale(1.02); }
    div[data-testid="stMetricValue"] { color: #00f260; font-size: 2rem !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    detector_general = YOLO('yolov8n.pt')
    
    yolo_path = 'runs/detect/smartvision_yolo/weights/best.pt'
    classifier_path = 'models/mobilenet_v2_smartvision.h5'
    
    detector_custom = YOLO(yolo_path) if os.path.exists(yolo_path) else detector_general
    
    classifier = None
    if os.path.exists(classifier_path):
        from tensorflow.keras.models import load_model
        classifier = load_model(classifier_path)
        
    return detector_general, detector_custom, classifier

try:
    with st.spinner("🚀 Initializing SmartVision Neural Core..."):
        detector_general, detector_custom, classifier = load_models()
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ SmartVision AI")
    st.caption("v4.0 | WebRTC Enabled")
    st.markdown("---")
    
    selected_page = st.radio(
        "SYSTEM MODULES",
        ["Dashboard", "Live Surveillance", "Driver Distract Analysis", "Diagnostics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Control Center")
    conf_threshold = st.slider("AI Confidence Threshold", 0.0, 1.0, 0.35)
    st.info("🟢 System Operational")

# ==========================================
# 4. MODULE: DASHBOARD (HOME)
# ==========================================
if selected_page == "Dashboard":
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>SmartVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; letter-spacing: 2px;'>ADVANCED SITUATIONAL AWARENESS PLATFORM</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("""<div class="neo-card"><h3>👁️ Omni-Watch</h3><p>Real-time surveillance.</p></div>""", unsafe_allow_html=True)
    with c2: st.markdown("""<div class="neo-card"><h3>🧠 Neuro-Guard</h3><p>Driver Behavior Analysis.</p></div>""", unsafe_allow_html=True)
    with c3: st.markdown("""<div class="neo-card"><h3>⚡ Live-Sync</h3><p>WebRTC Zero-Lag Streaming.</p></div>""", unsafe_allow_html=True)

# ==========================================
# 5. MODULE: LIVE SURVEILLANCE (WEBRTC ENABLED)
# ==========================================
elif selected_page == "Live Surveillance":
    st.title("🎥 Active Surveillance Feed")
    st.markdown("Secure Protocol: **WebRTC** (Browser Compatible)")
    
    # 1. STUN Configuration (Prevents Freezing/Black Screen)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # 2. Video Processor Class
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Run YOLO Detection
            results = detector_general(img, conf=conf_threshold)
            
            # Draw boxes
            annotated_frame = results[0].plot()
            
            # Return processed frame
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    col_video, col_stats = st.columns([3, 1])
    
    with col_stats:
        st.markdown("""<div class="neo-card"><h4>📡 Controls</h4></div>""", unsafe_allow_html=True)
        st.info("Click 'START' below to grant camera access.")
        
    with col_video:
        # 3. The WebRTC Component
        webrtc_streamer(
            key="smartvision-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# ==========================================
# 6. MODULE: DRIVER DISTRACT ANALYSIS
# ==========================================
elif selected_page == "Driver Distract Analysis":
    st.title("🚗 Driver Distract Analysis")
    
    st.markdown("""<div class="neo-card"><h4>📂 Evidence Ingestion</h4></div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Source (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    col_input_img, col_output_img = st.columns(2)

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        with col_input_img:
            st.image(image, caption="Original Source", use_column_width=True)
            
        st.write("") 
        if st.button("🚀 EXECUTE FORENSIC ANALYSIS"):
            with st.spinner("Analyzing..."):
                results = detector_custom(img_array, conf=conf_threshold)
                used_model = "Custom Core"
                if len(results[0].boxes) == 0:
                    results = detector_general(img_array, conf=0.15)
                    used_model = "General Core (Backup)"

                annotated_img = img_array.copy()
                final_status = "Safe Driving"
                is_danger = False
                
                detected_classes = [results[0].names[int(b.cls)] for b in results[0].boxes]
                distractions = ['cell phone', 'cup', 'bottle', 'remote', 'sandwich']
                has_distraction = any(x in distractions for x in detected_classes)

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cls_name = result.names[int(box.cls)]
                        
                        if cls_name in distractions:
                            label = f"THREAT: {cls_name.upper()}"
                            color = (255, 0, 0)
                            is_danger = True
                            final_status = "Distracted (Object Interaction)"
                        elif cls_name == 'person' and classifier:
                            if has_distraction:
                                label = "DISTRACTED (Object Confirmed)"
                                color = (255, 0, 0)
                                is_danger = True
                                final_status = "Distracted (Object Interaction)"
                            else:
                                crop = img_array[y1:y2, x1:x2]
                                if crop.size > 0:
                                    crop_resized = cv2.resize(crop, (224, 224)) / 255.0
                                    crop_input = np.expand_dims(crop_resized, axis=0)
                                    pred = classifier.predict(crop_input)
                                    sub_id = np.argmax(pred)
                                    CLASS_NAMES = ['Distracted', 'Safe Driving', 'Talking', 'Texting']
                                    if sub_id < len(CLASS_NAMES):
                                        sub_label = CLASS_NAMES[sub_id]
                                        if "Safe" not in sub_label:
                                            is_danger = True
                                            final_status = sub_label
                                        label = sub_label
                                        color = (0, 255, 0) if "Safe" in sub_label else (255, 0, 0)
                                    else: label = "Person"; color = (0, 255, 0)
                        else: label = cls_name.title(); color = (0, 255, 0)

                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(annotated_img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                with col_output_img:
                    st.image(annotated_img, caption=f"Analyzed Result ({used_model})", use_column_width=True)
                
                st.markdown("---")
                if is_danger:
                    st.markdown(f"""<div style="background: rgba(255, 75, 75, 0.1); border-left: 5px solid #ff4b4b; padding: 20px;"><h3>🚨 VIOLATION: {final_status}</h3></div>""", unsafe_allow_html=True)
                elif len(results[0].boxes) == 0:
                    st.warning("⚠️ Inconclusive Scan")
                else:
                    st.markdown("""<div style="background: rgba(0, 242, 96, 0.1); border-left: 5px solid #00f260; padding: 20px;"><h3>✅ COMPLIANT</h3></div>""", unsafe_allow_html=True)

# ==========================================
# 7. MODULE: DIAGNOSTICS
# ==========================================
elif selected_page == "Diagnostics":
    st.title("📊 System Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="neo-card"><h4>📈 Learning Curves</h4></div>""", unsafe_allow_html=True)
        if os.path.exists('runs/detect/smartvision_yolo/results.png'): st.image('runs/detect/smartvision_yolo/results.png', use_column_width=True)
        else: st.warning("No Log Data")
    with c2:
        st.markdown("""<div class="neo-card"><h4>🧩 Confusion Matrix</h4></div>""", unsafe_allow_html=True)
        if os.path.exists('runs/detect/smartvision_yolo/confusion_matrix.png'): st.image('runs/detect/smartvision_yolo/confusion_matrix.png', use_column_width=True)
        else: st.warning("No Log Data")
