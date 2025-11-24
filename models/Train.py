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
    dataset_location = None
    
    try:
        print("\n Connecting to Roboflow...")
        rf = Roboflow(api_key="uz3y4w6e6aydR1HYe2gb")
        print("loading Roboflow workspace...")
        project = rf.workspace("sudopi").project("iwwd-dumping-detection-ryvfu")
        print("loading Roboflow project...")
        
        # Check available versions
        print("\n Checking available dataset versions...")
        version_info = project.get_version_information()
        print(f"Available versions: {len(version_info)} found")
        
        # Display version details
        for v in version_info:
            v_num = v['id'].split('/')[-1]
            has_export = 'yolov8' in v.get('exports', [])
            splits = v.get('splits', {})
            print(f"  Version {v_num}: {v['images']} images, YOLOv8 export: {has_export}, Splits: {splits}")
        
        # Try versions in order: 4, 2
        for version_num in [4, 2]:
            try:
                print(f"\n Attempting to download version {version_num}...")
                version = project.version(version_num)
                dataset = version.download("yolov8")
                dataset_location = dataset.location
                print(f" Dataset downloaded/verified at: {dataset_location}")
                break
            except KeyboardInterrupt:
                print("\n⚠ Download interrupted by user. Exiting...")
                return
            except Exception as e:
                error_msg = str(e)
                print(f"⚠ Version {version_num} failed: {error_msg[:200]}")
                
                # Check if it's a network timeout or SSL error
                if "timeout" in error_msg.lower() or "ssl" in error_msg.lower():
                    print("  → Network/SSL error detected. Try again or check connection.")
                
                if version_num == 2:  # Last attempt
                    raise
                continue
        
        if not dataset_location:
            raise Exception("No valid dataset version could be downloaded")
            
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user")
        return
    except Exception as e:
        print(f"\n ERROR downloading dataset: {e}")
        print("\n Troubleshooting tips:")
        print("  1. Check your internet connection (stable connection required)")
        print("  2. Verify API key is valid on Roboflow dashboard")
        print("  3. Try running the script again (downloads may be cached)")
        print("  4. Check if version has YOLOv8 export enabled on Roboflow")
        print("  5. If timeout occurs, try downloading manually from Roboflow UI")
        return

    # 3. Load the Pre-trained Model
    print("\n Loading YOLOv8m (Medium) model...")
    try:
        model = YOLO('yolov8m.pt')
    except Exception as e:
        print(f" ERROR loading model: {e}")
        print("  → Model will be downloaded automatically on first run")
        return

    # 4. Configure Paths
    yaml_path = os.path.join(dataset_location, 'data.yaml')
    
    # Verify data.yaml exists
    if not os.path.exists(yaml_path):
        print(f"\n ERROR: data.yaml not found at {yaml_path}")
        print(f"  Dataset location contents: {os.listdir(dataset_location)}")
        return
    
    print(f"\n Starting training using config: {yaml_path}")
    
    # 5. Start Training
    try:
        results = model.train(
            # Dataset
            data=yaml_path,
            
            # Training Parameters
            epochs=80,
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
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n ERROR during training: {e}")
        print("  Check GPU memory, batch size, or dataset integrity")

if __name__ == '__main__':
    train_yolo_pipeline()