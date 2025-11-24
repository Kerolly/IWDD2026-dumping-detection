import os
from ultralytics import YOLO
from roboflow import Roboflow
import torch

def train_yolo_pipeline():
    """
    Main pipeline to download dataset from Roboflow and fine-tune YOLOv8.
    """
    print("---  Starting Training Pipeline ---")

    # 1. Check GPU availability
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Device selected: {device} ({torch.cuda.get_device_name(0) if device == '0' else 'CPU'})")

    # 2. Download Dataset from Roboflow
    try:
        print("\n Connecting to Roboflow...")
        rf = Roboflow(api_key="uz3y4w6e6aydR1HYe2gb")
        project = rf.workspace("sudopi").project("iwwd-dumping-detection-ryvfu")
        
        # Check available versions
        print("\n Checking available dataset versions...")
        try:
            version_info = project.get_version_information()
            print(f"Available versions: {version_info}")
        except:
            print("Could not retrieve version information")
        
        # Try version 4, fallback to version 2 if it doesn't exist
        try:
            print("\n Attempting to download version 4...")
            version = project.version(4)
        except Exception as e:
            print(f" Version 4 not found: {e}")
            print("Falling back to version 2...")
            version = project.version(2)
        
        dataset = version.download("yolov8")
        print(f" Dataset downloaded/verified at: {dataset.location}")
        
    except Exception as e:
        print(f" ERROR downloading dataset: {e}")
        print("\n Troubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. Verify API key is valid")
        print("  3. Confirm version 4 exists on Roboflow dashboard")
        return

    # 3. Load the Pre-trained Model
    print("\n Loading YOLOv8m (Medium) model...")
    model = YOLO('yolov8m.pt') 

    # 4. Configure Paths
    yaml_path = os.path.join(dataset.location, 'data.yaml')
    print(f"\n Starting training using config: {yaml_path}")
    
    # 5. Start Training
    results = model.train(
        # Dataset
        data=yaml_path,
        
        # Training Parameters
        epochs=20,
        imgsz=640,
        batch=16,
        
        # Optimization
        optimizer='Adam',
        lr0=0.001,
        patience=15,
        
        # Output & Saving
        project='training/runs',
        name='garbage_detect_v4',
        save=True,
        
        # Augmentation
        augment=True,
        fliplr=0.5,
        
        # System
        device=device,
        workers=8,
        verbose=True
    )

    print("\n Training Finished!")
    print(f" Results saved in: {results.save_dir}")
    print(f" Best model weights: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_yolo_pipeline()