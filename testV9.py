import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# CONFIGURATION (V9 - SIMPLIFIED FOR NEW/MOVING GARBAGE)
# ============================================================================
DEFAULT_GARBAGE_MODEL = '/root/IWDD2026/models/yolo_garbage_colab_v5.pt'
CONF_USE = 0.12  # Lower for better recall on new/moving garbage
IOU_THRESHOLD = 0.5
IMG_SIZE_MAIN = 1280

# Parameters (focus on new appearances and movements)
STATIONARY_DUR = 1.2  # Shorter to detect faster stops after move
RECENT_WIN_SEC = 0.9  # For recent movement check
MOVEMENT_TOL_BASE_PX = 40  # Low tol to detect subtle stationarity
MIN_MOVE_BEFORE_PX = 40  # Low to catch small moves (e.g., placed nearby)
MIN_TRACK_AGE_SEC = 0.25  # Short tracks OK for new items
MIN_MOVE_SOFT_BASE_PX = 12  # For small genuine motions (e.g., dumped on street)
AREA_MIN_BASE_PX2 = 250  # Anti-noise, but not too high
AREA_MAX_FRAC = 0.35  # Allow larger items

# Background filtering (only for truly static from start)
START_CHECK_WINDOW = 0.25  # Short init to catch more "new"
BG_MOVEMENT_TOL_BASE_PX = 25  # Filter only ultra-static

# Hysteresis (simple reset for moving)
MOVE_HIGH_FACTOR = 1.6
MOVE_RESET_SEC = 0.35

# Debug
DEBUG_ENABLED = True
DEBUG_SUBDIR_NAME = 'debug_v9'

# ============================================================================
# ARGUMENTS
# ============================================================================
def parse_arguments():
    p = argparse.ArgumentParser(description='IWDD v9 Inference - New/Moving Garbage Focus')
    p.add_argument('--videos', type=str, required=True, help='Input videos folder')
    p.add_argument('--results', type=str, required=True, help='Output results folder')
    return p.parse_args()

# ============================================================================
# HELPERS
# ============================================================================
def get_center_xywh(box_xywh):
    return (float(box_xywh[0]), float(box_xywh[1]))

def calculate_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def calculate_displacement(positions):
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.array(positions[-1]) - np.array(positions[0])))

def movement_over_recent(positions, frames_window):
    if len(positions) < 2:
        return 0.0
    n = max(2, frames_window)
    pts = positions[-n:]
    total = 0.0
    for i in range(1, len(pts)):
        total += float(np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1])))
    return total

# ============================================================================
# CORE PROCESSING PER VIDEO
# ============================================================================
def process_single_video(video_path, output_dir, model_garbage, debug=False):
    video_name = Path(video_path).stem
    output_txt = output_dir / f"{video_name}.txt"
    debug_dir = output_dir / DEBUG_SUBDIR_NAME
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg_lines = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        with open(output_txt, 'w'):
            pass
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()
    diag = float(np.hypot(W, H))
    scale = diag / 1500.0
    move_tol_low_px = MOVEMENT_TOL_BASE_PX * scale
    min_move_px = MIN_MOVE_BEFORE_PX * scale
    min_track_age_s = MIN_TRACK_AGE_SEC
    soft_path_px = MIN_MOVE_SOFT_BASE_PX * scale
    bg_move_tol_px = BG_MOVEMENT_TOL_BASE_PX * scale
    move_tol_high_px = move_tol_low_px * MOVE_HIGH_FACTOR
    move_reset_frames = int(round(MOVE_RESET_SEC * fps))
    frames_window = int(round(RECENT_WIN_SEC * fps))
    min_track_age_f = int(round(min_track_age_s * fps))
    area_min = AREA_MIN_BASE_PX2 * (scale ** 2)
    area_max = AREA_MAX_FRAC * (W * H)
    if debug:
        dbg_lines += [
            f"video={video_name}",
            f"fps={fps:.2f} W={W} H={H} scale={scale:.3f}",
            f"stationary_dur={STATIONARY_DUR:.2f} recent_win_sec={RECENT_WIN_SEC:.2f}",
            f"move_tol_low_px={move_tol_low_px:.1f} move_tol_high_px={move_tol_high_px:.1f}",
            f"min_move_px={min_move_px:.1f} soft_path_px={soft_path_px:.1f} min_track_age_s={min_track_age_s:.2f}",
            f"conf_use={CONF_USE:.2f} area_min={area_min:.1f} area_max={area_max:.1f}",
            ""
        ]
    tracks = {}  # tid -> state
    dbg_bg_filtered = 0
    dbg_new_appearance_accepts = 0
    dbg_was_moving_accepts = 0
    final_timestamp = None
    frame_count = 0
    results_stream = model_garbage.track(
        source=str(video_path),
        persist=True,
        conf=CONF_USE,
        iou=IOU_THRESHOLD,
        tracker="botsort.yaml",
        stream=True,
        verbose=False,
        imgsz=IMG_SIZE_MAIN
    )
    for res in results_stream:
        if final_timestamp is not None:
            break
        frame_count += 1
        current_time = frame_count / fps
        frame = res.orig_img
        if frame is None:
            continue
        for st in tracks.values():
            st['seen_this_frame'] = False
        if res.boxes is None or res.boxes.id is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xywh.cpu().numpy()
        tids = res.boxes.id.int().cpu().tolist()
        confs = res.boxes.conf.cpu().numpy().tolist()
        for box, tid, det_conf in zip(boxes, tids, confs):
            cx, cy, bw, bh = map(float, box)
            box_area = bw * bh
            if (box_area < area_min) or (box_area > area_max):
                continue
            center = (cx, cy)
            if tid not in tracks:
                tracks[tid] = {
                    'start_time': current_time,
                    'first_seen_frame': frame_count,
                    'positions': [],
                    'max_displacement': 0.0,
                    'path_len': 0.0,
                    'was_moving': False,
                    'stationary_start': None,
                    'moving_streak': 0,
                    'seen_in_init': False,
                    'bg_checked': False,
                    'is_background': False,
                    'last_seen_time': current_time,
                    'last_pos': center,
                    'appeared_after_init': (current_time > START_CHECK_WINDOW)
                }
            st = tracks[tid]
            st['seen_this_frame'] = True
            if current_time < START_CHECK_WINDOW:
                st['seen_in_init'] = True
            st['positions'].append(center)
            st['last_seen_time'] = current_time
            st['last_pos'] = center
            if len(st['positions']) >= 2:
                p1 = st['positions'][-2]
                p2 = st['positions'][-1]
                st['path_len'] += calculate_distance(p1, p2)
            # Background check (only if seen in init and low movement)
            if st['seen_in_init'] and not st['bg_checked'] and current_time >= (START_CHECK_WINDOW + 1.0 / fps):
                st['bg_checked'] = True
                disp_bg = calculate_displacement(st['positions'])
                if disp_bg < bg_move_tol_px:
                    st['is_background'] = True
                    dbg_bg_filtered += 1
            if st['is_background']:
                continue
            # Update movement
            disp_total = calculate_displacement(st['positions'])
            st['max_displacement'] = max(st['max_displacement'], disp_total)
            if max(st['path_len'], st['max_displacement']) > min_move_px:
                st['was_moving'] = True
            rmov = movement_over_recent(st['positions'], frames_window)
            is_stationary_now = (rmov < move_tol_low_px)
            is_moving_high = (rmov > move_tol_high_px)
            age_frames = frame_count - st['first_seen_frame']
            old_enough = (age_frames >= min_track_age_f)
            # Eligible only if new or moved (focus on your idea)
            eligible_by_move = st['was_moving']
            eligible_by_new = st['appeared_after_init']
            eligible_by_soft_path = (st['path_len'] > soft_path_px)
            eligible = is_stationary_now and old_enough and (eligible_by_move or eligible_by_new or eligible_by_soft_path)
            if eligible:
                if st['stationary_start'] is None:
                    st['stationary_start'] = current_time
                st['moving_streak'] = 0
                stationary_time = current_time - st['stationary_start']
                if stationary_time >= STATIONARY_DUR:
                    final_timestamp = st['stationary_start']
                    if eligible_by_new and not eligible_by_move:
                        dbg_new_appearance_accepts += 1
                    elif eligible_by_move:
                        dbg_was_moving_accepts += 1
                    break
            else:
                if st['stationary_start'] is not None:
                    if is_moving_high:
                        st['moving_streak'] += 1
                        if st['moving_streak'] >= move_reset_frames:
                            st['stationary_start'] = None
                            st['moving_streak'] = 0
                    else:
                        st['moving_streak'] = 0
    # Write output
    with open(output_txt, 'w') as f:
        if final_timestamp is not None:
            f.write(f"{final_timestamp:.2f}")
    # Debug log
    if debug:
        dbg_lines.append("")
        dbg_lines.append(f"final_timestamp={final_timestamp:.2f}" if final_timestamp is not None else "final_timestamp=None")
        dbg_lines.append(f"bg_filtered={dbg_bg_filtered}")
        dbg_lines.append(f"accepts_new_appearance={dbg_new_appearance_accepts} accepts_was_moving={dbg_was_moving_accepts}")
        dbg_path = debug_dir / f"{video_name}.log"
        with open(dbg_path, 'w') as df:
            df.write("\n".join(dbg_lines))

# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_arguments()
    input_dir = Path(args.videos)
    output_dir = Path(args.results)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== IWDD v9 Inference - Focus on New/Moving Garbage ===")
    print(f"Garbage model: {DEFAULT_GARBAGE_MODEL}")
    print(f"Videos: {input_dir}")
    print(f"Results: {output_dir}")
    print(f"CONF: {CONF_USE}")
    print(f"Debug logs: {'ON' if DEBUG_ENABLED else 'OFF'} -> {output_dir / DEBUG_SUBDIR_NAME}")
    print("===================================================")
    model_garbage = YOLO(DEFAULT_GARBAGE_MODEL)
    videos = sorted(input_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos. Starting...")
    for video_file in tqdm(videos, desc="Inference", unit="vid"):
        try:
            process_single_video(str(video_file), output_dir, model_garbage, debug=DEBUG_ENABLED)
        except Exception as e:
            with open(output_dir / f"{video_file.stem}.txt", 'w'):
                pass
            print(f"Error on {video_file.name}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()