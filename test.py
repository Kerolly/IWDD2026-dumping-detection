import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
# Path to trained YOLO model
MODEL_PATH = 'models/yolo_garbage_colab_v4.pt' 

# --- DETECTION PARAMETERS ---
CONF_THRESHOLD = 0.15        # Minimum confidence threshold for detections (15%)
IOU_THRESHOLD = 0.5          # Intersection over Union threshold for NMS
STATIONARY_DURATION = 0.5    # Minimum duration object must remain visible (seconds)
MOVEMENT_TOLERANCE = 80      # Maximum allowed movement in pixels (camera jitter tolerance)
ENABLE_DEBUG_VIDEO = True    # Enable debug video output with annotations

def parse_arguments():
    parser = argparse.ArgumentParser(description="IWDD 2026 Final Script")
    parser.add_argument('--videos', type=str, required=True, help='Path to input videos')
    parser.add_argument('--results', type=str, required=True, help='Path to output results')
    return parser.parse_args()

def calculate_movement(positions):
    """Calculate the Euclidean distance between initial and final positions."""
    if not positions:
        return 0.0
    start = np.array(positions[0])
    current = np.array(positions[-1])
    return np.linalg.norm(current - start)

def process_video(video_path, output_dir, model, debug_dir):
    video_name = Path(video_path).stem
    output_txt = output_dir / f"{video_name}.txt"
    
    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error opening {video_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize debug video writer (optional)
    video_writer = None
    if ENABLE_DEBUG_VIDEO:
        debug_path = debug_dir / f"{video_name}_debug.avi"
        video_writer = cv2.VideoWriter(str(debug_path), cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Initialize tracking data structure
    # Track ID -> {start_time, positions[], confirmed}
    tracks = defaultdict(lambda: {'start_time': None, 'positions': [], 'confirmed': False})
    
    final_timestamp = None
    frame_count = 0

    print(f"▶️ Processing: {video_name}...")

    # --- YOLO TRACKING ---
    # stream=True for memory efficiency, persist=True for consistent track IDs
    results = model.track(
        source=str(video_path),
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        tracker="botsort.yaml",
        stream=True,
        verbose=False
    )

    for result in results:
        frame_count += 1
        current_time = frame_count / fps
        frame = result.orig_img  # Cadrul original pentru desenat

        if result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x, y, w, h = box
                center = (float(x), float(y))
                
                track = tracks[track_id]
                
                # Initialize new track ID with current timestamp
                if track['start_time'] is None:
                    track['start_time'] = current_time
                
                # Update track position history
                track['positions'].append(center)
                
                # Calculate tracking metrics
                time_visible = current_time - track['start_time']
                movement = calculate_movement(track['positions'])
                
                # Default color: Yellow (tracking state)
                color = (0, 255, 255) 
                status_text = f"ID:{track_id} {time_visible:.1f}s Mov:{movement:.0f}px"

                # Evaluate dumping detection criteria
                # Check if object has been visible long enough AND has minimal movement
                if time_visible >= STATIONARY_DURATION:
                    if movement < MOVEMENT_TOLERANCE:
                        # Dumping event detected
                        color = (0, 0, 255) # Red
                        status_text = "DUMPING DETECTED!"
                        
                        if not track['confirmed']:
                            print(f"  🚨 ALERT: ID {track_id} dumping detected at {track['start_time']:.2f}s")
                            track['confirmed'] = True
                            if final_timestamp is None:
                                final_timestamp = track['start_time']
                    else:
                        # Object is moving (e.g., held by person walking)
                        color = (0, 255, 0) # Green
                        status_text = f"Moving ({movement:.0f}px)"

                # Draw debug annotations on frame
                if ENABLE_DEBUG_VIDEO:
                    # Draw bounding box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Draw status text
                    cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Write frame to debug video
        if video_writer:
            video_writer.write(frame)
            
        # Optional: Stop processing after first detection for performance
        # Comment out the lines below to process entire video for debug visualization
        if final_timestamp is not None and not ENABLE_DEBUG_VIDEO:
             break

    # Clean up resources
    cap.release()
    if video_writer:
        video_writer.release()

    # Write final results to file
    with open(output_txt, 'w') as f:
        if final_timestamp:
            f.write(f"{final_timestamp:.2f}")
            print(f"  ✅ Result saved: {final_timestamp:.2f}s")
        else:
            print(f"  ❌ No dumping detected.")

def main():
    args = parse_arguments()
    input_dir = Path(args.videos)
    output_dir = Path(args.results)
    debug_dir = Path("debug_output") # Directory for debug output videos
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_DEBUG_VIDEO:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    videos = list(input_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos.")

    for video_file in videos:
        process_video(video_file, output_dir, model, debug_dir)

if __name__ == "__main__":
    main()