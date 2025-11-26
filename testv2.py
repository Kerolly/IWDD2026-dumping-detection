import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
MODEL_PATH = 'models/yolo_garbage_colab_v2.pt' 

# --- LOGIC PARAMETERS (Calibrated) ---
CONF_THRESHOLD = 0.10        # Lower confidence to catch difficult objects
IOU_THRESHOLD = 0.5          # Tracking overlap
STATIONARY_DURATION = 1.0    # Must be still for 1s to be "dumped"
MOVEMENT_TOLERANCE = 80      # Pixels (jitter tolerance)
START_CHECK_WINDOW = 1.0     # How long to watch "start objects" to see if they are static

def parse_arguments():
    parser = argparse.ArgumentParser(description="IWDD 2026 Final Submission")
    parser.add_argument('--videos', type=str, required=True, help='Input video folder')
    parser.add_argument('--results', type=str, required=True, help='Output results folder')
    return parser.parse_args()

def calculate_total_displacement(positions):
    """Distance from First seen point to Last seen point."""
    if not positions: return 0.0
    start = np.array(positions[0])
    current = np.array(positions[-1])
    return np.linalg.norm(current - start)

def process_single_video(video_path, output_dir, model):
    video_name = Path(video_path).stem
    output_txt = output_dir / f"{video_name}.txt"
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Track Data: ID -> {start_time, positions[], is_background}
    track_history = defaultdict(lambda: {'start_time': None, 'positions': [], 'is_background': False})
    
    # IDs present in the very first frame
    start_frame_ids = set()
    
    final_timestamp = None
    frame_count = 0

    # --- INFERENCE ---
    results = model.track(
        source=str(video_path),
        persist=True,

        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        tracker="botsort.yaml",
        stream=True,
        verbose=False,

        imgsz=1280,

        augment=True
    )

    for result in results:
        if final_timestamp is not None: break

        frame_count += 1
        current_time = frame_count / fps

        if result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                
                # 1. Identify Objects present at T=0 (Potential Background)
                if frame_count == 1:
                    start_frame_ids.add(track_id)

                # 2. Update Tracking Logic
                center = (float(box[0]), float(box[1]))
                track = track_history[track_id]
                
                if track['start_time'] is None:
                    track['start_time'] = current_time
                
                track['positions'].append(center)
                
                # 3. Background Filtering Logic (The "Smart Filter")
                # If this ID was here at start, we check if it moves
                if track_id in start_frame_ids:
                    # Give it 'START_CHECK_WINDOW' seconds to prove it's moving
                    if current_time < START_CHECK_WINDOW:
                        continue # Keep watching
                    else:
                        # Time is up. Did it move?
                        initial_displacement = calculate_total_displacement(track['positions'])
                        
                        if not track.get('checked_background', False):
                            track['checked_background'] = True
                            # If it barely moved in the first second, it's a rock/old trash
                            if initial_displacement < MOVEMENT_TOLERANCE:
                                track['is_background'] = True
                            else:
                                # It moved! (e.g. falling at start). Treat as valid.
                                track['is_background'] = False
                
                # If marked as background, ignore forever
                if track['is_background']:
                    continue

                # 4. DUMPING DETECTION (Standard Rule)
                # Check if object is currently stationary for enough time
                time_visible = current_time - track['start_time']
                
                if time_visible >= STATIONARY_DURATION:
                    # Check displacement over the *entire* history to confirm dumping spot
                    # Or strictly check if it stabilized recently.
                    # Here we check total displacement from start.
                    # For a falling object: Start(Air) -> End(Ground) = High Displacement.
                    # For a placed object: Start(Hand) -> End(Ground) = High Displacement.
                    # Wait... we need to detect when it STOPS.
                    
                    # Check stability in the LAST 'STATIONARY_DURATION' seconds
                    # Get positions from N seconds ago
                    frames_needed = int(STATIONARY_DURATION * fps)
                    if len(track['positions']) > frames_needed:
                        recent_positions = track['positions'][-frames_needed:]
                        recent_movement = calculate_total_displacement(recent_positions)
                        
                        # If it stopped moving recently...
                        if recent_movement < MOVEMENT_TOLERANCE:
                            # ... but make sure it wasn't ALWAYS static (unless it's new)
                            # If it's a new ID (not start_id), just stopping is enough.
                            # If it's a start_id, we already verified it moved initially.
                            
                            final_timestamp = current_time - STATIONARY_DURATION
                            # Clamp to 0 if negative
                            if final_timestamp < 0: final_timestamp = 0.0
                            
                            print(f"  [DETECTED] {video_name}: ID {track_id} at {final_timestamp:.2f}s")
                            break

    cap.release()

    with open(output_txt, 'w') as f:
        if final_timestamp:
            f.write(f"Timestamp: {final_timestamp:.2f} s")

def main():
    args = parse_arguments()
    input_dir = Path(args.videos)
    output_dir = Path(args.results)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"[CRITICAL] Model not found: {MODEL_PATH}")
        return

    print(f"--- IWDD 2026 Engine V2 ---")
    model = YOLO(MODEL_PATH)
    
    videos = list(input_dir.glob("*.mp4"))
    for i, vid in enumerate(videos):
        print(f"Processing {i+1}/{len(videos)}: {vid.name}...", end='\r')
        process_single_video(vid, output_dir, model)
    
    print("\nDone.")

if __name__ == "__main__":
    main()