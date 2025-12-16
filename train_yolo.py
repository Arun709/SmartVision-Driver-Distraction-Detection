from ultralytics import YOLO
import os

def train_yolo():
    print("🚀 Starting YOLOv8 Training...")
    
    # 1. Load Pre-trained Model
    model = YOLO('yolov8n.pt')  # Nano version (Fastest)

    # 2. Train
    # Ensure 'data.yaml' is in the same folder where you run this script
    results = model.train(
        data='data.yaml',   
        epochs=30,          
        imgsz=640,
        batch=16,
        name='smartvision_yolo', 
        patience=10,         # Stop if no improvement for 10 epochs
        augment=True         # Use built-in augmentation
    )
    
    print("✅ YOLO Training Complete.")
    print("Best weights saved at: runs/detect/smartvision_yolo/weights/best.pt")

if __name__ == "__main__":
    if not os.path.exists("data.yaml"):
        print("❌ Error: data.yaml not found! Run the dataset download script first.")
    else:
        train_yolo()
