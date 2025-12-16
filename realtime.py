import cv2
from ultralytics import YOLO

def start_local_webcam():
    # 1. Load Model (YOLOv8 Nano - Fastest)
    print("⏳ Loading AI Model...")
    model = YOLO('yolov8n.pt') 
    print("✅ Model Loaded!")

    # 2. Open Webcam (Index 0 is usually default)
    cap = cv2.VideoCapture(0)

    # Set resolution (Lowering it makes it faster)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎥 Webcam started! Press 'Q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 3. Run Detection
        results = model(frame, conf=0.4, verbose=False)

        # 4. Draw boxes
        annotated_frame = results[0].plot()

        # 5. Show in a Desktop Window (Not Browser)
        cv2.imshow("SmartVision: Real-Time Detector (Press Q to Exit)", annotated_frame)

        # Press 'Q' to close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_local_webcam()
