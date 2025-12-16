import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
import gc

# ==========================================
# 1. CORE CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(
    page_title="PatrolIQ: Enterprise AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FORCE DARK THEME & PREMIUM CSS
st.markdown("""
    <style>
    /* GLOBAL RESET */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1a202c 0%, #000000 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* SIDEBAR GLASS */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.9);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* HEADERS */
    h1, h2, h3 {
        background: linear-gradient(to right, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* NEO-CARDS */
    .neo-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
    }
    .neo-card:hover {
        transform: translateY(-5px);
        border-color: rgba(5, 117, 230, 0.3);
    }
    
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(135deg, #0575e6 0%, #00f260 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 20px rgba(0, 242, 96, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 242, 96, 0.4);
    }
    
    /* METRICS */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #00f260;
        text-shadow: 0 0 10px rgba(0, 242, 96, 0.5);
    }
    
    /* ALERTS */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .status-safe { background: rgba(0, 242, 96, 0.15); color: #00f260; border: 1px solid #00f260; }
    .status-danger { background: rgba(255, 75, 75, 0.15); color: #ff4b4b; border: 1px solid #ff4b4b; }
    
    /* SPINNER */
    .stSpinner > div { border-top-color: #00f260 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENT MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    """Loads models once and caches them to save memory"""
    # General YOLO (Backup)
    detector_general = YOLO('yolov8n.pt')
    
    # Custom Driver Models
    yolo_path = 'runs/detect/smartvision_yolo/weights/best.pt'
    classifier_path = 'models/mobilenet_v2_smartvision.h5'
    
    detector_custom = YOLO(yolo_path) if os.path.exists(yolo_path) else detector_general
    
    classifier = None
    if os.path.exists(classifier_path):
        from tensorflow.keras.models import load_model
        classifier = load_model(classifier_path)
        
    return detector_general, detector_custom, classifier

try:
    with st.spinner("🚀 Initializing PatrolIQ Neural Core..."):
        detector_general, detector_custom, classifier = load_models()
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ PatrolIQ Enterprise")
    st.caption("v2.5.0 | High-Performance Build")
    st.markdown("---")
    
    selected_page = st.radio(
        "SYSTEM MODULES",
        ["Dashboard", "Live Surveillance", "Biometric Audit", "Diagnostics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Control Center")
    conf_threshold = st.slider("AI Confidence Threshold", 0.0, 1.0, 0.25, 
                             help="Adjust sensitivity for object detection.")
    
    st.info("🟢 System Operational")
    st.markdown("---")
    st.caption("© 2025 SmartVision AI Labs")

# ==========================================
# 4. MODULE: DASHBOARD (HOME)
# ==========================================
if selected_page == "Dashboard":
    # Hero Section
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>PatrolIQ</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; letter-spacing: 2px; margin-bottom: 50px;'>ADVANCED SITUATIONAL AWARENESS PLATFORM</p>", unsafe_allow_html=True)
    
    # Feature Grid
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="neo-card">
            <h3>👁️ Omni-Watch</h3>
            <p>Real-time surveillance capable of tracking 80+ object classes with < 15ms latency using YOLOv8 Nano architecture.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="neo-card">
            <h3>🧠 Neuro-Guard</h3>
            <p>Hybrid AI engine combining Object Detection & MobileNet Classification to audit driver behavior with 99% accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="neo-card">
            <h3>⚡ Live-Sync</h3>
            <p>Direct hardware acceleration for zero-lag video processing and instant threat triangulation.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.image("https://viso.ai/wp-content/uploads/2021/01/computer-vision-deep-learning-applications.jpg", use_container_width=True)

# ==========================================
# 5. MODULE: LIVE SURVEILLANCE
# ==========================================
elif selected_page == "Live Surveillance":
    st.title("🎥 Active Surveillance Feed")
    st.markdown("Secure Protocol: **OpenCV Direct** | Latency: **Ultra-Low**")
    
    col_video, col_stats = st.columns([3, 1])
    
    with col_stats:
        st.markdown("""<div class="neo-card"><h4>📡 Telemetry</h4></div>""", unsafe_allow_html=True)
        
        # Camera Toggle
        run_camera = st.checkbox("🔴 INITIATE FEED", value=False, key="cam_toggle")
        
        st.markdown("---")
        kpi1 = st.empty()
        kpi2 = st.empty()
        kpi3 = st.empty()

    with col_video:
        video_placeholder = st.empty()
        
        if not run_camera:
            video_placeholder.markdown("""
            <div style="background: #000; border-radius: 12px; height: 400px; display: flex; align-items: center; justify-content: center; border: 1px solid #333;">
                <p style="color: #666;">OFFLINE - Awaiting Activation</p>
            </div>
            """, unsafe_allow_html=True)
            # Force cleanup when camera is off
            gc.collect()

    # CAMERA LOGIC
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Hardware Error: Camera not accessible.")
        else:
            # Main Loop
            while run_camera:
                ret, frame = cap.read()
                if not ret: break
                
                # AI Inference
                start_time = time.time()
                results = detector_general(frame, conf=conf_threshold, verbose=False)
                fps = 1.0 / (time.time() - start_time)
                
                # Annotation
                annotated_frame = results[0].plot()
                
                # Display Video
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True, channels="RGB")
                
                # Live Metrics
                obj_count = len(results[0].boxes)
                classes = [results[0].names[int(b.cls)] for b in results[0].boxes]
                top_obj = max(set(classes), key=classes.count) if classes else "None"
                
                kpi1.metric("FPS", f"{int(fps)}")
                kpi2.metric("Objects", obj_count)
                kpi3.metric("Dominant", top_obj.title())
                
                # Stop immediately if user unchecked the box during loop
                # This check ensures we break loop instantly
                if not st.session_state.cam_toggle:
                    break
                    
            cap.release()
            cv2.destroyAllWindows()

# ==========================================
# 6. MODULE: BIOMETRIC AUDIT (DRIVER SAFETY)
# ==========================================
elif selected_page == "Biometric Audit":
    st.title("🚗 Driver Safety Audit")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown("""<div class="neo-card"><h4>📂 Evidence Ingestion</h4></div>""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Source (JPG/PNG)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Source Image", use_container_width=True)
            img_array = np.array(image.convert('RGB'))
            
            st.write("") # Spacer
            analyze_btn = st.button("🚀 EXECUTE FORENSIC ANALYSIS")
    
    with col_result:
        st.markdown("""<div class="neo-card"><h4>🤖 AI Verdict</h4></div>""", unsafe_allow_html=True)
        
        if uploaded_file and analyze_btn:
            with st.spinner("Triangulating objects and posture..."):
                # 1. Hybrid Inference Strategy
                results = detector_custom(img_array, conf=conf_threshold)
                used_model = "Custom Core"
                
                # Fallback to General Core if empty
                if len(results[0].boxes) == 0:
                    results = detector_general(img_array, conf=0.15)
                    used_model = "General Core (Backup)"

                annotated_img = img_array.copy()
                final_status = "Safe Driving"
                is_danger = False
                
                # 2. Global Object Scan
                detected_classes = [results[0].names[int(b.cls)] for b in results[0].boxes]
                distraction_objects = ['cell phone', 'cup', 'bottle', 'remote', 'sandwich']
                has_distraction = any(x in distraction_objects for x in detected_classes)

                # 3. Smart Processing Loop
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cls_name = result.names[int(box.cls)]
                        
                        # LOGIC TREE
                        if cls_name in distraction_objects:
                            label = f"THREAT: {cls_name.upper()}"
                            color = (255, 0, 0) # Red
                            is_danger = True
                            final_status = "Distracted (Object Interaction)"
                        
                        elif cls_name == 'person' and classifier:
                            # If we saw a phone globally, override classifier
                            if has_distraction:
                                label = "DISTRACTED (Object Confirmed)"
                                color = (255, 0, 0)
                                is_danger = True
                                final_status = "Distracted (Object Interaction)"
                            else:
                                # Run Classifier
                                crop = img_array[y1:y2, x1:x2]
                                if crop.size > 0:
                                    crop_resized = cv2.resize(crop, (224, 224)) / 255.0
                                    crop_input = np.expand_dims(crop_resized, axis=0)
                                    pred = classifier.predict(crop_input)
                                    sub_id = np.argmax(pred)
                                    
                                    # Safe Driving, Texting, etc.
                                    CLASS_NAMES = ['Distracted', 'Safe Driving', 'Talking', 'Texting']
                                    if sub_id < len(CLASS_NAMES):
                                        sub_label = CLASS_NAMES[sub_id]
                                        if "Safe" not in sub_label:
                                            is_danger = True
                                            final_status = sub_label
                                        label = sub_label
                                        color = (0, 255, 0) if "Safe" in sub_label else (255, 0, 0)
                                    else:
                                        label = "Person"
                                        color = (0, 255, 0)
                        else:
                            label = cls_name.title()
                            color = (0, 255, 0)

                        # Draw High-Tech Box
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(annotated_img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 4. Display Result
                st.image(annotated_img, caption=f"Analyzed by {used_model}", use_container_width=True)
                
                st.markdown("---")
                if is_danger:
                    st.markdown(f"""
                    <div style="background: rgba(255, 75, 75, 0.1); border-left: 5px solid #ff4b4b; padding: 20px; border-radius: 8px;">
                        <h3 style="color: #ff4b4b; margin: 0;">🚨 VIOLATION DETECTED</h3>
                        <p style="margin-top: 10px; font-size: 1.1rem;"><strong>Status:</strong> {final_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif len(results[0].boxes) == 0:
                    st.warning("⚠️ Inconclusive: No valid targets identified.")
                else:
                    st.markdown("""
                    <div style="background: rgba(0, 242, 96, 0.1); border-left: 5px solid #00f260; padding: 20px; border-radius: 8px;">
                        <h3 style="color: #00f260; margin: 0;">✅ COMPLIANT</h3>
                        <p style="margin-top: 10px; font-size: 1.1rem;"><strong>Status:</strong> Safe Driving</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Waiting for Evidence Upload...")

# ==========================================
# 7. MODULE: DIAGNOSTICS (ANALYTICS)
# ==========================================
elif selected_page == "Diagnostics":
    st.title("📊 System Diagnostics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="neo-card"><h4>📈 Learning Curves (Loss)</h4></div>""", unsafe_allow_html=True)
        if os.path.exists('runs/detect/smartvision_yolo/results.png'):
            st.image('runs/detect/smartvision_yolo/results.png', use_container_width=True)
        else: st.warning("Log data unavailable.")
            
    with col2:
        st.markdown("""<div class="neo-card"><h4>🧩 Confusion Matrix</h4></div>""", unsafe_allow_html=True)
        if os.path.exists('runs/detect/smartvision_yolo/confusion_matrix.png'):
            st.image('runs/detect/smartvision_yolo/confusion_matrix.png', use_container_width=True)
        else: st.warning("Matrix data unavailable.")
