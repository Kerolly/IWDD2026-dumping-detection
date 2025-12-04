import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
#                               CONFIGURATION (V6)
# ============================================================================

# --- Paths ---
DEFAULT_MODEL_PATH = 'models/yolo_garbage_colab_v3.pt'
DEFAULT_AGENT_MODEL = 'yolov8n.pt'   # COCO model for agents (person, vehicles)

# --- Detection Thresholds ---
CONF_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5

# --- Temporal Logic ---
STATIONARY_DURATION   = 1.40   # seconds required stationary to confirm dump
MOVEMENT_TOLERANCE    = 55     # base pixels (scaled by frame diagonal)
RECENT_WINDOW_SEC     = 0.8    # movement window to determine "stationary"
MIN_MOVE_BEFORE_DUMP  = 70     # base pixels (scaled): must have moved this much before stopping
MIN_TRACK_AGE         = 0.4    # seconds: minimum age before eligible
START_CHECK_WINDOW    = 0.4    # seconds: ignore/init background filtering

# --- Interaction Logic ---
PROXIMITY_THRESHOLD   = 160    # pixels: if an agent is closer, pause stationary timer

# COCO agent classes: 0:Person, 2:Car, 3:Motorcycle, 5:Bus, 7:Truck
AGENT_CLASSES = [0, 2, 3, 5, 7]


# ============================================================================
#                             HELPER FUNCTIONS
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="IWDD 2026 Inference Engine V6")
    parser.add_argument('--videos', type=str, required=True, help='Path to input videos')
    parser.add_argument('--results', type=str, required=True, help='Path to output results')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to custom garbage model')
    return parser.parse_args()


def get_center(box_xywh):
    return (float(box_xywh[0]), float(box_xywh[1]))


def calculate_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calculate_displacement(positions):
    if not positions or len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.array(positions[-1]) - np.array(positions[0])))


def recent_movement(positions, frames_window):
    if len(positions) < 2:
        return 0.0
    n = max(2, frames_window)
    pts = positions[-n:]
    total = 0.0
    for i in range(1, len(pts)):
        total += float(np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1])))
    return total


# ============================================================================
#                             CORE LOGIC
# ============================================================================

def process_single_video(video_path, output_dir, model_garbage, model_agents):
    video_name = Path(video_path).stem
    output_txt = output_dir / f"{video_name}.txt"

    # Read video metadata for fps and shape scaling
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # Create empty file to mark evaluated with no detection
        with open(output_txt, 'w'):
            pass
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()

    # Scale pixel thresholds by frame diagonal to be resolution-agnostic
    diag = float(np.hypot(w, h))
    scale = diag / 1500.0  # 1500 ~ diag of 1280x720
    move_tol_px = MOVEMENT_TOLERANCE * scale
    min_move_px = MIN_MOVE_BEFORE_DUMP * scale
    frames_window = int(round(RECENT_WINDOW_SEC * fps))
    min_track_age_frames = int(round(MIN_TRACK_AGE * fps))

    # Track state
    track_history = defaultdict(lambda: {
        'start_time': None,
        'positions': [],
        'was_moving': False,
        'max_displacement': 0.0,
        'stationary_start': None,
        'seen_in_init': False,
        'bg_checked': False,
        'is_background': False,
        'first_seen_frame': None
    })

    final_timestamp = None
    frame_count = 0

    # Streaming tracking over the entire video (stable IDs)
    results_stream = model_garbage.track(
        source=str(video_path),
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        tracker="botsort.yaml",
        stream=True,
        verbose=False,
        imgsz=1280
    )

    for res in results_stream:
        if final_timestamp is not None:
            break

        frame_count += 1
        current_time = frame_count / fps

        frame = res.orig_img  # BGR frame
        if frame is None:
            # Safety: skip if no frame
            continue

        # Detect agents on current frame
        agent_centers = []
        agents_res = model_agents(frame, verbose=False, conf=0.25, classes=AGENT_CLASSES)
        if len(agents_res) and agents_res[0].boxes is not None:
            boxes_a = agents_res[0].boxes.xywh.cpu().numpy()
            agent_centers = [get_center(b) for b in boxes_a]

        # No garbage detections this frame
        if res.boxes is None or res.boxes.id is None:
            continue

        boxes = res.boxes.xywh.cpu().numpy()
        track_ids = res.boxes.id.int().cpu().tolist()

        for box, tid in zip(boxes, track_ids):
            center = get_center(box)
            track = track_history[tid]

            # Initialize track
            if track['start_time'] is None:
                track['start_time'] = current_time
                track['first_seen_frame'] = frame_count

            # Update positions history
            track['positions'].append(center)

            # Initial background window
            if current_time < START_CHECK_WINDOW:
                track['seen_in_init'] = True
                # Continue accumulating history but do not trigger any decision
                continue

            # Background check once after init window
            if track['seen_in_init'] and not track['bg_checked']:
                track['bg_checked'] = True
                if calculate_displacement(track['positions']) < move_tol_px:
                    track['is_background'] = True

            if track['is_background']:
                continue

            # Update movement history
            disp = calculate_displacement(track['positions'])
            track['max_displacement'] = max(track['max_displacement'], disp)
            if track['max_displacement'] > min_move_px:
                track['was_moving'] = True

            # Recent movement to decide stationary
            rmov = recent_movement(track['positions'], frames_window)
            is_stationary_now = (rmov < move_tol_px)

            # Proximity to agents pauses timer
            is_agent_near = False
            if agent_centers:
                nearest = min(calculate_distance(center, ac) for ac in agent_centers)
                is_agent_near = nearest < PROXIMITY_THRESHOLD

            # Age guard
            track_age = frame_count - (track['first_seen_frame'] or frame_count)
            old_enough = track_age >= min_track_age_frames

            # Start/maintain/reset stationary timer
            if is_stationary_now and track['was_moving'] and not is_agent_near and old_enough:
                if track['stationary_start'] is None:
                    track['stationary_start'] = current_time

                stationary_time = current_time - track['stationary_start']
                if stationary_time >= STATIONARY_DURATION:
                    final_timestamp = track['stationary_start']
                    break
            else:
                track['stationary_start'] = None

    # Write output (empty file if no detection)
    with open(output_txt, 'w') as f:
        if final_timestamp is not None:
            f.write(f"{final_timestamp:.2f}")


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

    print("--- IWDD 2026 Engine V6 (Streaming + Stationary State Machine) ---")
    print(f"Model:   {args.model}")
    print(f"Videos:  {input_dir}")
    print(f"Results: {output_dir}")

    # Load models
    print("Loading models...")
    model_garbage = YOLO(args.model)
    model_agents = YOLO(DEFAULT_AGENT_MODEL)
    print("Models loaded. Starting inference...")

    videos = sorted(input_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos")

    for video_file in tqdm(videos, desc="Inference", unit="vid"):
        try:
            process_single_video(str(video_file), output_dir, model_garbage, model_agents)
        except Exception as e:
            # Always create an empty file on error so it's counted as evaluated
            with open(output_dir / f"{video_file.stem}.txt", 'w'):
                pass
            print(f"Error on {video_file.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()