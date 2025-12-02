"""
IWDD 2026 - YOLOv8 Professional Training & Evaluation Script
Garbage Detection Model (Fine-Tuning)

Pipeline:
1. Data Download (Roboflow) & Cleanup
2. Training (Fine-tuning with AdamW)
3. Final Evaluation (Validation & Test Sets)
"""

import os
import sys
import cv2
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")

# ============================================================================
#                              CONFIGURATION
# ============================================================================

# --- Roboflow Configuration ---
API_KEY = 'uz3y4w6e6aydR1HYe2gb' 
WORKSPACE = 'sudopi'
PROJECT = 'iwdd-dumping-detection-kpzg9'
VERSION = 1 

# --- Model Settings ---
MODEL = 'yolov8m.pt'    # Medium model
EPOCHS = 100            # Maximum epochs
PATIENCE = 25           # Early stopping threshold

# --- Hardware Settings ---
BATCH = 16              # Conservative batch size. Increase to 24/32 if VRAM allows.
IMGSZ = 640             # Standard resolution
WORKERS = 4             # Set to 0 if you encounter multiprocessing errors on Windows
DEVICE = 0              # GPU ID
CACHE = 'ram'

# --- Optimization (Fine-Tuning Strategy) ---
OPTIMIZER = 'AdamW'     # Optimized for convergence
LR0 = 0.0005            # Low initial learning rate for stability
LRF = 0.01              # Final learning rate decay
COS_LR = True           # Cosine annealing
WARMUP_EPOCHS = 3       
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1

# --- Regularization ---
DROPOUT = 0.15          # Prevent overfitting
WEIGHT_DECAY = 0.0005   

# --- Data Augmentation ---
AUGMENT = True
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4             # High brightness variance for CCTV day/night
DEGREES = 15.0
TRANSLATE = 0.1
SCALE = 0.5
MOSAIC = 1.0
MIXUP = 0.15
COPY_PASTE = 0.0

# --- Output Paths ---
SAVE_DIR = './training_results'


# ============================================================================
#                              UTILITY FUNCTIONS
# ============================================================================

def install_requirements():
    """Checks and installs required Python packages."""
    import subprocess
    packages = ['ultralytics', 'roboflow', 'opencv-python', 'tqdm', 'pandas', 'seaborn']
    print("Checking dependencies...")
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_').split('[')[0])
        except ImportError:
            print(f"Installing missing package: {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

def download_dataset():
    """Downloads the dataset from Roboflow."""
    from roboflow import Roboflow
    print("\n--- Downloading Dataset ---")
    
    # Attempt to clear cache
    cache_path = Path.home() / '.cache' / 'roboflow'
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
        except:
            pass
    
    rf = Roboflow(api_key=API_KEY)
    proj = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = proj.version(VERSION).download("yolov8")
    
    print(f"Dataset path: {dataset.location}")
    return dataset.location

def cleanup_images(dataset_path):
    """
    Validates image integrity. Deletes corrupt files or converts non-JPG formats.
    Crucial for preventing OpenCV errors during training.
    """
    print("\n--- validating & Cleaning Dataset ---")
    dataset_path = Path(dataset_path)
    stats = {'checked': 0, 'fixed': 0, 'removed': 0}
    
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        if not split_path.exists(): continue
        
        # Normalize directory structure
        images_dir = split_path / 'images' if (split_path / 'images').exists() else split_path
        labels_dir = split_path / 'labels' if (split_path / 'labels').exists() else split_path
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            image_files.extend(images_dir.glob(ext))
        
        print(f"Scanning {split}: {len(image_files)} images...")

        for img_path in image_files:
            stats['checked'] += 1
            try:
                # Verify read
                img = cv2.imread(str(img_path))
                if img is None:
                    # Attempt binary read
                    with open(img_path, 'rb') as f:
                        img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                
                if img is None:
                    # Delete corrupt
                    img_path.unlink()
                    label = labels_dir / f"{img_path.stem}.txt"
                    if label.exists(): label.unlink()
                    stats['removed'] += 1
                    continue
                
                # Convert to standard JPG to fix header issues
                new_path = img_path.with_suffix('.jpg')
                cv2.imwrite(str(new_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if img_path.suffix.lower() != '.jpg' and img_path.exists():
                    img_path.unlink()
                
                stats['fixed'] += 1
                
            except Exception:
                # Force delete on error
                try:
                    img_path.unlink()
                    label = labels_dir / f"{img_path.stem}.txt"
                    if label.exists(): label.unlink()
                except: pass
                stats['removed'] += 1
    
    # Remove cached labels
    for cache in dataset_path.rglob('*.cache'):
        cache.unlink()
    
    print(f"Cleanup: Checked={stats['checked']}, Fixed={stats['fixed']}, Removed={stats['removed']}")

# ============================================================================
#                              TRAINING ENGINE
# ============================================================================

def train_model(dataset_path):
    """Executes the YOLOv8 training loop."""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING PHASE")
    print("=" * 60)
    
    model = YOLO(MODEL)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"yolo_garbage_{timestamp}"
    
    print(f"Run Name: {run_name}")
    print(f"GPU: {DEVICE} | Batch: {BATCH} | LR: {LR0}")
    print("-" * 60)
    
    results = model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        amp=True,           # Automatic Mixed Precision
        cache='ram',
        
        # Optimization
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        cos_lr=COS_LR,
        
        # Regularization
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
        
        # Augmentation
        augment=AUGMENT,
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
        degrees=DEGREES, translate=TRANSLATE, scale=SCALE,
        mosaic=MOSAIC, mixup=MIXUP, copy_paste=COPY_PASTE,
        
        # Output
        project=SAVE_DIR,
        name=run_name,
        save=True,
        plots=True,
        verbose=True,
        
    )
    
    return run_name

# ============================================================================
#                              EVALUATION ENGINE
# ============================================================================

def run_final_evaluation(run_name, dataset_path):
    """
    Loads the best model from training and runs validation on the TEST set.
    Prints detailed metrics to the console.
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("STARTING FINAL EVALUATION (TEST SET)")
    print("=" * 60)
    
    # Path to the best weights
    best_weights = Path(SAVE_DIR) / run_name / 'weights' / 'best.pt'
    
    if not best_weights.exists():
        print(f"CRITICAL: Best model not found at {best_weights}")
        return

    print(f"Loading best model: {best_weights}")
    best_model = YOLO(best_weights)
    
    # 1. Validate on 'test' split
    # Note: we specify split='test' to use unseen data
    print("\nRunning inference on TEST split...")
    metrics = best_model.val(
        data=f"{dataset_path}/data.yaml",
        split='test',
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=SAVE_DIR,
        name=f"{run_name}_TEST_EVAL"
    )
    
    # 2. Extract and Print Metrics
    print("\n" + "-" * 40)
    print("FINAL MODEL PERFORMANCE REPORT")
    print("-" * 40)
    
    # Extract metrics safely
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    
    print(f"Precision (Confidence):  {precision:.4f}")
    print(f"Recall (Sensitivity):    {recall:.4f}")
    print(f"mAP@50 (Standard):       {map50:.4f}")
    print(f"mAP@50-95 (Robustness):  {map50_95:.4f}")
    print("-" * 40)
    
    # 3. Location info
    print(f"\nLogs and Confusion Matrix saved at:")
    print(f"{Path(SAVE_DIR) / run_name}_TEST_EVAL")
    print("\nDone.")

# ============================================================================
#                              MAIN EXECUTION
# ============================================================================

def main():
    print("IWDD 2026 - Local Training & Eval Pipeline")
    
    install_requirements()
    dataset_path = download_dataset()
    cleanup_images(dataset_path)
    
    try:
        # 1. Train
        run_name = train_model(dataset_path)
        
        # 2. Evaluate
        run_final_evaluation(run_name, dataset_path)
        
        print("\nALL TASKS COMPLETED SUCCESSFULLY.")
        
    except KeyboardInterrupt:
        print("\n[!] Process interrupted.")
    except Exception as e:
        print(f"\n[!] Critical Failure: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == '__main__':
    main()