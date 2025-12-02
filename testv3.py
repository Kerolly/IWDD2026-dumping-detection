import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm  # Ensure tqdm is installed: pip install tqdm

# ============================================================================
#                               CONFIGURATION
# ============================================================================

# --- Default Paths (Local Structure) ---
DEFAULT_MODEL_PATH = 'models/yolo_garbage_colab_v4.pt' 
DEFAULT_SAFETY_MODEL = 'yolov8n.pt'   # Standard YOLOv8 Nano (auto-downloads)

# --- Detection Thresholds ---
CONF_THRESHOLD = 0.20        # Minimum confidence to track a garbage object
IOU_THRESHOLD = 0.5          # NMS Intersection over Union threshold

# --- Temporal Logic (The "Brain") ---
STATIONARY_DURATION = 1.0    # Time (seconds) an object must remain still to be classified as dumping
MOVEMENT_TOLERANCE = 80      # Maximum pixel displacement allowed for a "stationary" object
START_CHECK_WINDOW = 0.3     # Time window at video start to filter out pre-existing background objects

# --- Safety Filter Settings ---
ENABLE_SAFETY_FILTER = True  # Enable cross-reference with standard YOLO model
SAFETY_IOU_THRESHOLD = 0.30  # Max allowed overlap with a safety object (30%)
# COCO Classes to filter out (False Positives):
# 0=Person, 1=Bicycle, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck, 15=Cat, 16=Dog
SAFETY_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16] 


# ============================================================================
#                             HELPER FUNCTIONS
# ============================================================================

def parse_arguments():
    """
    Parses command-line arguments for local execution.
    """
    parser = argparse.ArgumentParser(description="IWDD 2026 Inference Engine (Local)")
    
    parser.add_argument('--videos', type=str, required=True, 
                        help='Path to input video directory (e.g., data/videos)')
    
    parser.add_argument('--results', type=str, required=True, 
                        help='Path to output results directory (e.g., results)')
    
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, 
                        help='Path to the fine-tuned garbage detection model (.pt file)')

    return parser.parse_args()


def calculate_displacement(positions):
    """
    Calculates the Euclidean distance between the first and last tracked position.
    
    Args:
        positions (list): List of (x, y) coordinates.
    Returns:
        float: Distance in pixels.
    """
    if not positions:
        return 0.0
    start = np.array(positions[0])
    current = np.array(positions[-1])
    return np.linalg.norm(current - start)


def calculate_intersection_ratio(box_a, box_b):
    """
    Calculates the overlap ratio between the candidate garbage box (A) 
    and a safety object box (B).
    
    Formula: Intersection Area / Area of Box A
    This checks if the garbage detection is largely contained within a person/vehicle.
    """
    # Helper to convert xywh center to xyxy corners
    def get_corners(b):
        x, y, w, h = b
        return x - w/2, y - h/2, x + w/2, y + h/2

    ax1, ay1, ax2, ay2 = get_corners(box_a)
    bx1, by1, bx2, by2 = get_corners(box_b)

    # Determine intersection rectangle
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    
    if area_a == 0: return 0.0
    
    return intersection_area / area_a


# ============================================================================
#                             CORE LOGIC
# ============================================================================

def process_single_video(video_path, output_dir, model_garbage, model_safety):
    """
    Processes a single video file to detect illegal dumping events using logic + dual models.
    """
    video_name = Path(video_path).stem
    output_txt = output_dir / f"{video_name}.txt"
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # State tracking data structure
    # ID -> {'start_time': float, 'positions': [], 'is_background': bool, 'checked': bool}
    track_history = defaultdict(lambda: {
        'start_time': None, 
        'positions': [], 
        'is_background': False, 
        'checked_background': False
    })
    
    start_frame_ids = set()
    final_timestamp = None
    frame_count = 0

    # --- Processing Loop ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Optimization: Stop processing immediately if an event is confirmed
        if final_timestamp is not None:
            break

        frame_count += 1
        current_time = frame_count / fps

        # 1. Primary Inference: Garbage Detection + Tracking
        # Using imgsz=1280 to improve small object detection logic
        results = model_garbage.track(
            frame, 
            persist=True, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD, 
            tracker="botsort.yaml", 
            verbose=False, 
            imgsz=1280 
        )

        # 2. Secondary Inference: Safety Filter (Person/Vehicle Detection)
        safety_boxes = []
        if ENABLE_SAFETY_FILTER and results[0].boxes.id is not None:
            # Run inference only (no tracking needed for safety check)
            safety_results = model_safety(frame, verbose=False, conf=0.25, classes=SAFETY_CLASSES)
            if len(safety_results) > 0:
                safety_boxes = safety_results[0].boxes.xywh.cpu().numpy()

        # 3. Process Garbage Tracks
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                
                # --- SAFETY FILTER CHECK ---
                # Check if this garbage candidate overlaps with a safety object (Human/Car)
                if ENABLE_SAFETY_FILTER and len(safety_boxes) > 0:
                    is_false_positive = False
                    for s_box in safety_boxes:
                        if calculate_intersection_ratio(box, s_box) > SAFETY_IOU_THRESHOLD:
                            is_false_positive = True
                            break
                    if is_false_positive:
                        continue # Skip processing this object, it is likely a person
                # ---------------------------

                # Calculate Center
                center = (float(box[0]), float(box[1]))
                track = track_history[track_id]
                
                # Initialize Track
                if track['start_time'] is None:
                    track['start_time'] = current_time
                
                track['positions'].append(center)
                
                # --- BACKGROUND FILTERING (Smart Filter) ---
                # If object exists at Frame 1, verify if it moves or is static background
                if frame_count == 1:
                    start_frame_ids.add(track_id)
                
                if track_id in start_frame_ids:
                    if current_time < START_CHECK_WINDOW:
                        continue # Still monitoring start behavior
                    else:
                        if not track['checked_background']:
                            disp = calculate_displacement(track['positions'])
                            track['checked_background'] = True
                            
                            # If minimal movement in first second, assume static background
                            if disp < MOVEMENT_TOLERANCE:
                                track['is_background'] = True
                            else:
                                track['is_background'] = False
                
                if track['is_background']:
                    continue

                # --- DUMPING DETECTION LOGIC ---
                time_visible = current_time - track['start_time']
                
                if time_visible >= STATIONARY_DURATION:
                    # Logic: Check stability in the LAST 'STATIONARY_DURATION' seconds
                    frames_needed = int(STATIONARY_DURATION * fps)
                    
                    if len(track['positions']) > frames_needed:
                        # Analyze recent movement history
                        recent_positions = track['positions'][-frames_needed:]
                        recent_movement = calculate_displacement(recent_positions)
                        
                        if recent_movement < MOVEMENT_TOLERANCE:
                            # EVENT DETECTED: Object appeared, moved (or was new), and then stopped.
                            final_timestamp = current_time - STATIONARY_DURATION
                            if final_timestamp < 0: final_timestamp = 0.0
                            
                            # print(f"  [DETECTED] {video_name} | ID {track_id} | Time: {final_timestamp:.2f}s")
                            break

    cap.release()

    # Write output timestamp to file
    with open(output_txt, 'w') as f:
        if final_timestamp is not None:
            f.write(f"{final_timestamp:.2f}")


# ============================================================================
#                             MAIN ENTRY POINT
# ============================================================================

def main():
    args = parse_arguments()
    input_dir = Path(args.videos)
    output_dir = Path(args.results)
    model_path = args.model
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[CRITICAL] Model not found at: {model_path}")
        print("Please check the path or download your trained model.")
        return

    print("--- IWDD 2026 Inference Engine (Local) ---")
    print(f"Garbage Model: {model_path}")
    print(f"Safety Model:  {DEFAULT_SAFETY_MODEL}")
    
    # Load Models
    # Garbage model: Your custom trained weights
    model_garbage = YOLO(model_path)
    # Safety model: Pre-trained YOLOv8n (detects person, car, etc.)
    model_safety = YOLO(DEFAULT_SAFETY_MODEL)
    
    # Get list of videos
    videos = list(input_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos. Starting inference...")
    
    # Process videos with progress bar
    for video_file in tqdm(videos, desc="Processing", unit="vid"):
        process_single_video(video_file, output_dir, model_garbage, model_safety)
    
    print("\nProcessing Complete. Results saved.")

if __name__ == "__main__":
    main()