import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
#                               CONFIGURATION
# ============================================================================

# --- Paths ---
DEFAULT_MODEL_PATH = 'models/yolo_garbage_colab_v4.pt'
# We need standard YOLO to detect "Agents" (People, Cars, Trucks)
DEFAULT_AGENT_MODEL = 'yolov8n.pt'   

# --- Detection Thresholds (Stricter to reduce FP) ---
CONF_THRESHOLD = 0.35        # Increased from 0.20 to 0.35 (Trust only strong detections)
IOU_THRESHOLD = 0.5          

# --- Temporal Logic (Slower to reduce FP) ---
STATIONARY_DURATION = 2.5    # Object must be abandoned for 2.5s (Filters brief stops)
MOVEMENT_TOLERANCE = 150     # Increased to 150px (Handles wind/camera shake better)
START_CHECK_WINDOW = 1.0     # Time to classify pre-existing objects

# --- Interaction Logic (The "Smart" Brain) ---
PROXIMITY_THRESHOLD = 250    # Pixels. If Agent is closer than this, timer is PAUSED.
SPAWN_RADIUS = 350           # Pixels. New trash MUST spawn within this radius of an Agent.
                             # If it spawns far away, it's likely a "Ghost" or Error.

# COCO Classes for Agents: 0:Person, 2:Car, 3:Motorcycle, 5:Bus, 7:Truck
AGENT_CLASSES = [0, 2, 3, 5, 7]

# ============================================================================
#                             HELPER FUNCTIONS
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="IWDD 2026 Inference Engine V4")
    parser.add_argument('--videos', type=str, required=True, help='Path to input videos')
    parser.add_argument('--results', type=str, required=True, help='Path to output results')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to custom model')
    return parser.parse_args()

def get_center(box):
    """Returns center (x, y) from xywh box."""
    return (float(box[0]), float(box[1]))

def calculate_distance(p1, p2):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_displacement(positions):
    """Total displacement from start to end of history."""
    if not positions: return 0.0
    return np.linalg.norm(np.array(positions[-1]) - np.array(positions[0]))

# ============================================================================
#                             CORE LOGIC
# ============================================================================

def process_single_video(video_path, output_dir, model_garbage, model_agents):
    video_name = Path(video_path).stem
    output_txt = output_dir / f"{video_name}.txt"
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Track Data Structure
    # ID -> {
    #   'start_time': float, 
    #   'positions': [], 
    #   'abandoned_start': float|None, 
    #   'is_background': bool, 
    #   'is_ghost': bool
    # }
    track_history = defaultdict(lambda: {
        'start_time': None, 
        'positions': [], 
        'abandoned_start': None, 
        'is_background': False,
        'checked_bg': False,
        'is_ghost': False,
        'spawn_checked': False
    })
    
    start_frame_ids = set()
    final_timestamp = None
    frame_count = 0

    # --- MAIN LOOP ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        if final_timestamp is not None: break # Optimization: Stop after detection

        frame_count += 1
        current_time = frame_count / fps

        # 1. Detect Agents (Context Layer)
        # We find all People and Cars in the frame
        agent_centers = []
        results_agents = model_agents(frame, verbose=False, conf=0.25, classes=AGENT_CLASSES)
        if len(results_agents) > 0 and results_agents[0].boxes is not None:
            boxes_a = results_agents[0].boxes.xywh.cpu().numpy()
            for box in boxes_a:
                agent_centers.append(get_center(box))

        # 2. Detect & Track Garbage (Target Layer)
        # Using imgsz=1280 for small objects logic
        results_g = model_garbage.track(
            frame, persist=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, 
            tracker="botsort.yaml", verbose=False, imgsz=1280
        )

        if results_g[0].boxes.id is not None:
            boxes = results_g[0].boxes.xywh.cpu().numpy()
            track_ids = results_g[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                center = get_center(box)
                track = track_history[track_id]
                
                # --- A. INITIALIZATION & SPAWN CHECK ---
                if track['start_time'] is None:
                    track['start_time'] = current_time
                    track['positions'].append(center)
                    
                    # Logic: If object appears late (not at start), it MUST appear near an Agent.
                    # If it appears alone in the middle of nowhere, it's a False Positive (Ghost).
                    if frame_count > 30: # Give 1s buffer for video start
                        min_dist = float('inf')
                        if agent_centers:
                            dists = [calculate_distance(center, ac) for ac in agent_centers]
                            min_dist = min(dists)
                        
                        if min_dist > SPAWN_RADIUS:
                            track['is_ghost'] = True # Mark as invalid
                            # print(f"Ghost ignored: ID {track_id} (Dist: {min_dist})")
                        else:
                            track['spawn_checked'] = True # Valid spawn
                    else:
                        # Objects at T=0 are valid candidates (subject to background check)
                        track['spawn_checked'] = True

                # Update position
                if len(track['positions']) == 0: track['positions'].append(center)
                else: track['positions'].append(center)

                # Skip invalid tracks
                if track['is_ghost']: continue

                # --- B. BACKGROUND FILTER (Pre-existing Objects) ---
                if frame_count == 1: start_frame_ids.add(track_id)
                
                if track_id in start_frame_ids:
                    if current_time < START_CHECK_WINDOW: continue
                    if not track['checked_bg']:
                        track['checked_bg'] = True
                        # If it barely moved since start, it's background
                        if calculate_displacement(track['positions']) < MOVEMENT_TOLERANCE:
                            track['is_background'] = True
                
                if track['is_background']: continue

                # --- C. INTERACTION & ABANDONMENT ---
                # Check distance to nearest agent
                is_handled = False
                if agent_centers:
                    dists = [calculate_distance(center, ac) for ac in agent_centers]
                    if min(dists) < PROXIMITY_THRESHOLD:
                        is_handled = True
                
                # Logic: 
                # If handled -> Reset Timer (Timer = None)
                # If alone -> Start Timer
                
                if is_handled:
                    track['abandoned_start'] = None 
                else:
                    if track['abandoned_start'] is None:
                        track['abandoned_start'] = current_time

                # --- D. DUMPING CONFIRMATION ---
                if track['abandoned_start'] is not None:
                    time_abandoned = current_time - track['abandoned_start']
                    
                    if time_abandoned >= STATIONARY_DURATION:
                        # Final check: Was it actually stationary during the abandonment?
                        frames_needed = int(time_abandoned * fps)
                        if len(track['positions']) > frames_needed:
                            recent_pos = track['positions'][-frames_needed:]
                            movement = calculate_displacement(recent_pos)
                            
                            if movement < MOVEMENT_TOLERANCE:
                                # BINGO!
                                final_timestamp = track['abandoned_start']
                                if final_timestamp < 0: final_timestamp = 0.0
                                break

    cap.release()
    
    # Write Output
    with open(output_txt, 'w') as f:
        if final_timestamp: f.write(f"{final_timestamp:.2f}")

# ============================================================================
#                             MAIN ENTRY
# ============================================================================

def main():
    args = parse_arguments()
    input_dir = Path(args.videos)
    output_dir = Path(args.results)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.model):
        print(f"CRITICAL: Garbage model not found at {args.model}")
        return

    print("--- IWDD 2026 Engine V4 (Proximity & Spawn Logic) ---")
    
    # Load Models
    print("Loading Models...")
    model_garbage = YOLO(args.model)
    model_agents = YOLO(DEFAULT_AGENT_MODEL) 
    
    videos = list(input_dir.glob("*.mp4"))
    print(f"Processing {len(videos)} videos...")
    
    for video_file in tqdm(videos, desc="Inference"):
        process_single_video(video_file, output_dir, model_garbage, model_agents)
    
    print("\nDone.")

if __name__ == "__main__":
    main()