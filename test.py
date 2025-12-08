import cv2
import os
import argparse
import random
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION (V9.6)
# ============================================================================

DEFAULT_GARBAGE_MODEL = 'model_v5.pt'
CONF_USE = 0.15
IOU_THRESHOLD = 0.5
IMG_SIZE_MAIN = 1280

# Movement / timing
STATIONARY_DUR = 1.2
RECENT_WIN_SEC = 0.9
MOVEMENT_TOL_BASE_PX = 40
MIN_MOVE_BEFORE_PX = 40
MIN_TRACK_AGE_SEC = 0.25
MIN_MOVE_SOFT_BASE_PX = 12

# Area gating
RECENT_A_SEC = 0.8
LOOKBACK_A_SEC = 0.8
AREA_JUMP_RATIO = 1.10
AREA_CV_MIN = 0.10

# Near-cluster
CLUSTER_RADIUS_BASE = 350
CLUSTER_MIN_AGE_SEC = 0.8

# Long-term static background
BG_STATIC_MAX_SEC = 1.5
PERSIST_AFTER_SEC = 0.40

# Area noise
AREA_MIN_BASE_PX2 = 250
AREA_MAX_FRAC = 0.35

# Background at start
START_CHECK_WINDOW = 0.25
BG_MOVEMENT_TOL_BASE_PX = 25

# Hysteresis
MOVE_HIGH_FACTOR = 1.6
MOVE_RESET_SEC = 0.35

# Optical flow
FLOW_MAG_THRESH = 1.0
FLOW_WIN_SEC = 0.5
FLOW_COND_PATH_THRESH = 20

# Dual BG params
SHORT_HISTORY = 50
SHORT_VAR_THRESH = 25
LONG_HISTORY = 1000
LONG_VAR_THRESH = 16
DF_RATIO_THRESH = 0.3
FG_RATIO_THRESH = 0.2

# Frame skipping
SKIP_FRAMES = 2

# ============================================================================
# HELPERS
# ============================================================================
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

def median_or_zero(vals):
    return float(np.median(vals)) if len(vals) else 0.0

# ============================================================================
# CORE LOGIC
# ============================================================================
def analyze_video(video_path, model_garbage):
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()

    diag = float(np.hypot(W, H))
    scale = diag / 1500.0

    # Scaled parameters
    move_tol_low_px   = MOVEMENT_TOL_BASE_PX * scale
    min_move_px       = MIN_MOVE_BEFORE_PX * scale
    min_track_age_s   = MIN_TRACK_AGE_SEC
    soft_path_px      = MIN_MOVE_SOFT_BASE_PX * scale
    bg_move_tol_px    = BG_MOVEMENT_TOL_BASE_PX * scale
    move_tol_high_px  = move_tol_low_px * MOVE_HIGH_FACTOR
    move_reset_frames = int(round(MOVE_RESET_SEC * fps))
    frames_window     = int(round(RECENT_WIN_SEC * fps))
    min_track_age_f   = int(round(min_track_age_s * fps))
    area_min          = AREA_MIN_BASE_PX2 * (scale ** 2)
    area_max          = AREA_MAX_FRAC * (W * H)
    recent_f          = int(round(RECENT_A_SEC * fps))
    back_f            = int(round(LOOKBACK_A_SEC * fps))
    cluster_radius    = CLUSTER_RADIUS_BASE * scale
    flow_frames       = int(round(FLOW_WIN_SEC * fps))
    flow_cond_thresh  = FLOW_COND_PATH_THRESH * scale

    # Init dual BG
    short_sub = cv2.createBackgroundSubtractorMOG2(history=SHORT_HISTORY, varThreshold=SHORT_VAR_THRESH, detectShadows=False)
    long_sub = cv2.createBackgroundSubtractorMOG2(history=LONG_HISTORY, varThreshold=LONG_VAR_THRESH, detectShadows=True)
    prev_gray = None

    tracks = {}  # tid -> state
    bg_clusters = [] 

    final_timestamp = None
    frame_count = 0
    cv_count = 0

    # YOLO Streaming
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

    # Functions
    def area_medians_and_cv(area_list):
        n = len(area_list)
        if n < (recent_f + back_f + 2):
            return None, None, None
        
        recent = area_list[-recent_f:]
        prev   = area_list[-(recent_f + back_f):-recent_f]
        if len(prev) == 0 or len(recent) == 0:
            return None, None, None
        
        med_recent = median_or_zero(recent)
        med_prev   = median_or_zero(prev)
        mean_r = float(np.mean(recent))
        std_r  = float(np.std(recent)) if mean_r > 1e-9 else 0.0
        cv_r   = (std_r / (mean_r + 1e-9))
        return med_recent, med_prev, cv_r

    def compute_optical_flow(gray_frame, prev_gray, positions, flow_win):
        if prev_gray is None or len(positions) < flow_win:
            return 0.0
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = 0.0
        count_pts = 0

        for pos in positions[-flow_win:]:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < mag.shape[1] and 0 <= y < mag.shape[0]:
                avg_mag += mag[y, x]
                count_pts += 1

        return avg_mag / max(1, count_pts)

    # Frames processing
    for res in results_stream:
        if final_timestamp is not None:
            break
        
        frame_count += 1
        current_time = frame_count / fps

        frame = res.orig_img
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        do_cv = (cv_count % SKIP_FRAMES == 0)
        cv_count += 1

        if do_cv:
            fg_short = short_sub.apply(gray)
            fg_long = long_sub.apply(gray)
            df_mask = (fg_long > 0) & (fg_short == 0)
        else:
            fg_long = np.zeros_like(gray)
            df_mask = np.zeros_like(gray)

        # Reset seen flags
        for st in tracks.values():
            st['seen_this_frame'] = False

        if res.boxes is None or res.boxes.id is None or len(res.boxes) == 0:
            if do_cv:
                prev_gray = gray.copy()
            continue

        boxes = res.boxes.xywh.cpu().numpy()
        tids  = res.boxes.id.int().cpu().tolist()
        
        for box, tid in zip(boxes, tids):
            cx, cy, bw, bh = map(float, box)
            box_area = bw * bh
            if (box_area < area_min) or (box_area > area_max):
                continue
            center = (cx, cy)
            x1, y1, x2, y2 = int(cx - bw/2), int(cy - bh/2), int(cx + bw/2), int(cy + bh/2)

            # CV checks
            is_foreground = False
            eligible_by_dual_bg = False
            if do_cv:
                # bounds check
                h_img, w_img = gray.shape
                x1, x2 = max(0, x1), min(w_img, x2)
                y1, y2 = max(0, y1), min(h_img, y2)
                
                if x2 > x1 and y2 > y1:
                    roi_fg = fg_long[y1:y2, x1:x2]
                    fg_ratio = np.sum(roi_fg > 0) / (roi_fg.size + 1e-9)
                    is_foreground = fg_ratio > FG_RATIO_THRESH

                    roi_df = df_mask[y1:y2, x1:x2]
                    df_ratio = np.sum(roi_df) / (roi_df.size + 1e-9)
                    eligible_by_dual_bg = df_ratio > DF_RATIO_THRESH

            if tid not in tracks:
                tracks[tid] = {
                    'start_time': current_time,
                    'first_seen_frame': frame_count,
                    'positions': [],
                    'areas': [],
                    'max_displacement': 0.0,
                    'path_len': 0.0,
                    'was_moving': False,
                    'stationary_start': None,
                    'moving_streak': 0,
                    'seen_in_init': False,
                    'bg_checked': False,
                    'is_background': False,
                    'last_seen_time': current_time,
                    'appeared_after_init': (current_time > START_CHECK_WINDOW),
                    'cluster_added': False
                }
            
            st = tracks[tid]
            st['seen_this_frame'] = True
            if current_time < START_CHECK_WINDOW:
                st['seen_in_init'] = True
            
            st['positions'].append(center)
            st['areas'].append(box_area)
            
            if len(st['positions']) >= 2:
                p1 = st['positions'][-2]
                p2 = st['positions'][-1]
                st['path_len'] += calculate_distance(p1, p2)

            # BG logic
            if st['seen_in_init'] and not st['bg_checked'] and current_time >= (START_CHECK_WINDOW + 1.0/fps):
                st['bg_checked'] = True
                disp_bg = calculate_displacement(st['positions'])
                if disp_bg < bg_move_tol_px:
                    st['is_background'] = True
                    bg_clusters.append({'pos': center, 'created_at': current_time})
            
            if st['is_background']:
                continue

            # Movement Logic
            disp_total = calculate_displacement(st['positions'])
            st['max_displacement'] = max(st['max_displacement'], disp_total)

            if max(st['path_len'], st['max_displacement']) > min_move_px:
                st['was_moving'] = True
            
            rmov = movement_over_recent(st['positions'], frames_window)
            is_stationary_now = (rmov < move_tol_low_px)
            is_moving_high = (rmov > move_tol_high_px)
            old_enough = ((frame_count - st['first_seen_frame']) >= min_track_age_f)

            # Advanced Logic (Novelty, Flow, Clusters)
            med_r, med_p, cv_r = area_medians_and_cv(st['areas'])
            eligible_by_area_jump = False
            if (med_r is not None) and (med_p is not None) and (med_p > 0):
                ratio = med_r / (med_p + 1e-9)

                if (ratio >= AREA_JUMP_RATIO) or (cv_r is not None and cv_r >= AREA_CV_MIN and med_r > med_p):
                    eligible_by_area_jump = True
            
            near_cluster = False

            if st['appeared_after_init'] and bg_clusters:
                dists = [calculate_distance(center, c['pos']) for c in bg_clusters]
                if dists and min(dists) <= cluster_radius:
                    near_cluster = True
            
            eligible_by_flow = False

            if do_cv and st['path_len'] < flow_cond_thresh:
                flow_mag = compute_optical_flow(gray, prev_gray, st['positions'], flow_frames)
                eligible_by_flow = (flow_mag > FLOW_MAG_THRESH)

            # Static pile -> Background
            if (not st['was_moving']) and (not near_cluster) and (st['path_len'] <= soft_path_px) and \
               (not eligible_by_area_jump) and (not eligible_by_flow) and (not is_foreground) and (not eligible_by_dual_bg):
                
                if is_stationary_now and (current_time - st['start_time'] >= BG_STATIC_MAX_SEC):
                    st['is_background'] = True
                    bg_clusters.append({'pos': center, 'created_at': current_time})
                    continue

            eligible_by_move = st['was_moving']
            eligible_by_new  = st['appeared_after_init']
            eligible_by_soft = (st['path_len'] > soft_path_px)
            
            novelty_ok = (eligible_by_soft or eligible_by_area_jump or near_cluster or eligible_by_flow or is_foreground or eligible_by_dual_bg)
            
            eligible = (is_stationary_now and old_enough and (eligible_by_move or (eligible_by_new and novelty_ok)))

            # Add cluster point
            if (st['stationary_start'] is not None) and (not st['cluster_added']):
                if (current_time - st['stationary_start']) >= CLUSTER_MIN_AGE_SEC:
                    bg_clusters.append({'pos': center, 'created_at': current_time})
                    st['cluster_added'] = True

            if eligible:
                if st['stationary_start'] is None:
                    st['stationary_start'] = current_time
                st['moving_streak'] = 0
                
                stationary_time = current_time - st['stationary_start']
                if stationary_time >= (STATIONARY_DUR + PERSIST_AFTER_SEC):
                    final_timestamp = st['stationary_start']
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
        
        if do_cv:
            prev_gray = gray.copy()
            
    return final_timestamp

# ============================================================================
# TEMPLATE STRUCTURE
# ============================================================================

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_parameter()

    # Initialize the model
    # ------------------------------------------------------------------
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    print(f"Loading model from: {DEFAULT_GARBAGE_MODEL}")
    model_garbage = YOLO(DEFAULT_GARBAGE_MODEL)
    print("Model loaded. Starting inference...")
    # ------------------------------------------------------------------

    # For all the test videos
    video_list = sorted(os.listdir(args.videos)) 
    
    for video_file in video_list:
        full_video_path = os.path.join(args.videos, video_file)
        
        # Skip non-video files
        if not (video_file.lower().endswith('.mp4') or video_file.lower().endswith('.avi') or video_file.lower().endswith('.mov')):
            continue

        print(f"Processing: {video_file}")

        # Processing the video
        detected_timestamp = analyze_video(full_video_path, model_garbage)

        # Output file construction
        output_filename = video_file + ".txt" # Respectam strict template-ul
        output_path = os.path.join(args.results, output_filename)

        f = open(output_path, "w")
        
        # Writing the results
        if detected_timestamp is not None:
            f.write(f"{detected_timestamp:.2f}")
        else:
            # If there is nothing, will be an empty file
            pass 
        
        f.close()
    
    print("Done.")