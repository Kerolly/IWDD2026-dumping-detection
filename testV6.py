import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import statistics

# ============================================================================
#                               CONFIGURATION (HYBRID)
# ============================================================================

# Models (set your paths here)
DEFAULT_GARBAGE_MODEL = 'models/yolo_garbage_colab_v3.pt'  # change if needed
DEFAULT_AGENT_MODEL   = 'yolov8n.pt'                       # COCO agents (person/vehicles)

# Detection thresholds
CONF_THRESHOLD = 0.20
IOU_THRESHOLD  = 0.5
IMG_SIZE_MAIN  = 1280

# Mode decision (per-video)
MODE_WINDOW_SEC       = 1.0    # analyze first N seconds for blur/agents
BLUR_THRESH_FALLBACK  = 80.0   # <= fallback to object-only
AGENT_RATIO_MIN       = 0.10   # if < fallback
AGENT_CONF_MED_MIN    = 0.22   # if < fallback

# Agent-aware parameters (good video)
AA_STATIONARY_DURATION   = 1.4
AA_RECENT_WINDOW_SEC     = 0.8
AA_MOVEMENT_TOL_BASE_PX  = 55
AA_MIN_MOVE_BEFORE_PX    = 70
AA_MIN_TRACK_AGE_SEC     = 0.4
PROXIMITY_THRESHOLD_BASE = 160  # pause timer if agent within this radius (scaled)

# Object-only parameters (fallback on blur)
OF_STATIONARY_DURATION   = 1.6
OF_RECENT_WINDOW_SEC     = 1.0
OF_MOVEMENT_TOL_BASE_PX  = 70
OF_MIN_MOVE_BEFORE_PX    = 70
OF_MIN_TRACK_AGE_SEC     = 0.6

# Hysteresis
MOVE_HIGH_FACTOR     = 1.5     # high threshold = low * factor
MOVE_RESET_SEC       = 0.3     # need > this time above high to reset stationary

# Background filtering
START_CHECK_WINDOW   = 0.5     # seconds from video start considered init

# Bridging IDs (handle ID switches)
BRIDGE_RADIUS_BASE_PX = 60     # scaled by diag
BRIDGE_TIME_SEC       = 0.5

# Smoothing (EMA) on center positions
EMA_ALPHA = 0.6                # 0..1, higher = more weight on current

# ROI Zoom (only on blur/low conf)
ROI_ENABLE         = True
ROI_SCALE          = 1.6
ROI_IMGSZ          = 960
ROI_CONF_TRIGGER   = 0.16
ROI_COOLDOWN_SEC   = 1.0       # per-track cooldown

# Debug control
DEBUG_ENABLED       = False     # set True to save per-video debug logs
DEBUG_SUBDIR_NAME   = 'debug'   # results/<DEBUG_SUBDIR_NAME>/

# COCO agent classes: 0:Person, 2:Car, 3:Motorcycle, 5:Bus, 7:Truck
AGENT_CLASSES = [0, 2, 3, 5, 7]


# ============================================================================
#                               ARGUMENTS
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="IWDD Hybrid Inference (Agent-aware vs Object-only with blur fallback)")
    parser.add_argument('--videos', type=str, required=True, help='Input videos folder')
    parser.add_argument('--results', type=str, required=True, help='Output results folder')
    return parser.parse_args()


# ============================================================================
#                               HELPERS
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

def laplacian_blur_score(gray_frame):
    # higher variance = sharper, lower = blur
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def expand_and_crop(frame, x, y, w, h, scale):
    H, W = frame.shape[:2]
    new_w = w * scale
    new_h = h * scale
    x1 = int(clamp(x - new_w / 2, 0, W - 1))
    y1 = int(clamp(y - new_h / 2, 0, H - 1))
    x2 = int(clamp(x + new_w / 2, 0, W - 1))
    y2 = int(clamp(y + new_h / 2, 0, H - 1))
    roi = frame[y1:y2, x1:x2].copy()
    return roi, x1, y1

def median_or_zero(vals):
    try:
        return float(statistics.median(vals)) if vals else 0.0
    except statistics.StatisticsError:
        return 0.0


# ============================================================================
#                    MODE DECISION (BLUR / AGENT COVERAGE)
# ============================================================================

def decide_mode(video_path, model_agents, mode_window_sec, agent_classes):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {'mode': 'object', 'blur_med': 0.0, 'agent_ratio': 0.0, 'agent_conf_med': 0.0, 'frames': 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(round(mode_window_sec * fps))
    frames_checked = 0

    blur_scores = []
    agent_frames = 0
    agent_confs = []

    while frames_checked < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_checked += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(laplacian_blur_score(gray))

        # agent detection
        if model_agents is not None:
            ra = model_agents(frame, verbose=False, conf=0.25, classes=agent_classes)
            if len(ra) and ra[0].boxes is not None and len(ra[0].boxes) > 0:
                agent_frames += 1
                agent_confs.extend(ra[0].boxes.conf.cpu().numpy().tolist())

    cap.release()

    blur_med = median_or_zero(blur_scores)
    agent_ratio = (agent_frames / frames_checked) if frames_checked > 0 else 0.0
    agent_conf_med = median_or_zero(agent_confs)

    # fallback conditions
    fallback = (blur_med <= BLUR_THRESH_FALLBACK) or (agent_ratio < AGENT_RATIO_MIN) or (agent_conf_med < AGENT_CONF_MED_MIN)

    return {
        'mode': 'object' if fallback else 'agent',
        'blur_med': blur_med,
        'agent_ratio': agent_ratio,
        'agent_conf_med': agent_conf_med,
        'frames': frames_checked,
        'fps': fps
    }


# ============================================================================
#                        ROI RE-DETECTION (OPTIONAL)
# ============================================================================

def refine_with_roi(model_garbage, frame, box_xywh, roi_conf=ROI_CONF_TRIGGER, roi_imgsz=ROI_IMGSZ, scale=ROI_SCALE):
    x, y, w, h = box_xywh
    roi, x1, y1 = expand_and_crop(frame, x, y, w, h, scale)
    if roi.size == 0:
        return None

    rr = model_garbage(roi, conf=roi_conf, imgsz=roi_imgsz, verbose=False)
    if not len(rr) or rr[0].boxes is None or len(rr[0].boxes) == 0:
        return None

    # take highest confidence detection
    confs = rr[0].boxes.conf.cpu().numpy()
    boxes = rr[0].boxes.xywh.cpu().numpy()
    idx = int(np.argmax(confs))
    cx, cy, rw, rh = boxes[idx]
    refined_center = (float(cx) + x1, float(cy) + y1)
    refined_conf = float(confs[idx])
    return refined_center, refined_conf


# ============================================================================
#                          CORE PROCESSING PER VIDEO
# ============================================================================

def process_single_video(video_path, output_dir, model_garbage, model_agents, debug=False):
    video_name = Path(video_path).stem
    output_txt = output_dir / f"{video_name}.txt"

    # Debug dir under results
    debug_dir = output_dir / DEBUG_SUBDIR_NAME
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg_lines = []

    # Read basic metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        with open(output_txt, 'w'):
            pass
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()

    # Decide mode
    mode_info = decide_mode(video_path, model_agents, MODE_WINDOW_SEC, AGENT_CLASSES)
    mode = mode_info['mode']   # 'agent' or 'object'
    blur_med = mode_info.get('blur_med', 0.0)
    agent_ratio = mode_info.get('agent_ratio', 0.0)
    agent_conf_med = mode_info.get('agent_conf_med', 0.0)

    # Scale thresholds to frame diagonal
    diag = float(np.hypot(W, H))
    scale = diag / 1500.0

    if mode == 'agent':
        stationary_dur   = AA_STATIONARY_DURATION
        recent_win_sec   = AA_RECENT_WINDOW_SEC
        move_tol_low_px  = AA_MOVEMENT_TOL_BASE_PX * scale
        min_move_px      = AA_MIN_MOVE_BEFORE_PX * scale
        min_track_age_s  = AA_MIN_TRACK_AGE_SEC
        proximity_thresh = PROXIMITY_THRESHOLD_BASE * scale
        use_proximity    = True
    else:
        stationary_dur   = OF_STATIONARY_DURATION
        recent_win_sec   = OF_RECENT_WINDOW_SEC
        move_tol_low_px  = OF_MOVEMENT_TOL_BASE_PX * scale
        min_move_px      = OF_MIN_MOVE_BEFORE_PX * scale
        min_track_age_s  = OF_MIN_TRACK_AGE_SEC
        proximity_thresh = None
        use_proximity    = False

    move_tol_high_px   = move_tol_low_px * MOVE_HIGH_FACTOR
    move_reset_frames  = int(round(MOVE_RESET_SEC * fps))
    frames_window      = int(round(recent_win_sec * fps))
    min_track_age_f    = int(round(min_track_age_s * fps))
    bridge_radius_px   = BRIDGE_RADIUS_BASE_PX * scale
    bridge_time_s      = BRIDGE_TIME_SEC

    # Debug header
    if debug:
        dbg_lines.append(f"video={video_name}")
        dbg_lines.append(f"mode={mode} blur_med={blur_med:.1f} agent_ratio={agent_ratio:.2f} agent_conf_med={agent_conf_med:.2f}")
        dbg_lines.append(f"fps={fps:.2f} W={W} H={H} scale={scale:.3f}")
        dbg_lines.append(f"stationary_dur={stationary_dur:.2f} recent_win_sec={recent_win_sec:.2f}")
        dbg_lines.append(f"move_tol_low_px={move_tol_low_px:.1f} move_tol_high_px={move_tol_high_px:.1f}")
        dbg_lines.append(f"min_move_px={min_move_px:.1f} min_track_age_s={min_track_age_s:.2f}")
        dbg_lines.append(f"proximity={'ON' if use_proximity else 'OFF'} prox_thresh={proximity_thresh if proximity_thresh else 0:.1f}")
        dbg_lines.append("")

    # Canonical tracks (handle ID switches)
    alias_map = {}         # tid -> canon_id
    canon_tracks = {}      # canon_id -> state
    orphans = []           # list of dicts: {canon_id, last_pos, last_time}

    # Stats for debug
    dbg_bg_filtered = 0
    dbg_bridges = 0
    dbg_roi_calls = 0
    dbg_new_appearance_accepts = 0
    dbg_was_moving_accepts = 0

    final_timestamp = None
    frame_count = 0

    # Streaming tracking for garbage
    results_stream = model_garbage.track(
        source=str(video_path),
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        tracker="botsort.yaml",
        stream=True,
        verbose=False,
        imgsz=IMG_SIZE_MAIN
    )

    def get_canon_state(canon_id, now_time, now_frame, init_pos):
        if canon_id not in canon_tracks:
            canon_tracks[canon_id] = {
                'canon_id': canon_id,
                'start_time': now_time,
                'first_seen_frame': now_frame,
                'positions_raw': [],
                'positions_smooth': [],
                'last_smooth': None,
                'max_displacement': 0.0,
                'was_moving': False,
                'stationary_start': None,
                'moving_streak': 0,   # frames above move_high
                'seen_in_init': False,
                'bg_checked': False,
                'is_background': False,
                'last_seen_time': now_time,
                'last_pos': init_pos,
                'roi_last_time': -1e9,
                'appeared_after_init': (now_time > START_CHECK_WINDOW)
            }
        return canon_tracks[canon_id]

    def bridge_find_orphan(pos, now_time):
        nonlocal orphans
        best_idx = -1
        best_dist = 1e18
        for i, o in enumerate(orphans):
            dt = now_time - o['last_time']
            if dt < 0 or dt > bridge_time_s:
                continue
            d = calculate_distance(pos, o['last_pos'])
            if d < best_dist and d <= bridge_radius_px:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            entry = orphans.pop(best_idx)
            return entry['canon_id']
        return None

    for res in results_stream:
        if final_timestamp is not None:
            break

        frame_count += 1
        current_time = frame_count / fps
        frame = res.orig_img
        if frame is None:
            continue

        # Mark all canon tracks unseen at frame start
        for st in canon_tracks.values():
            st['seen_this_frame'] = False

        # Agents for proximity (only in agent-aware mode)
        agent_centers = []
        if use_proximity and model_agents is not None:
            ra = model_agents(frame, verbose=False, conf=0.25, classes=AGENT_CLASSES)
            if len(ra) and ra[0].boxes is not None and len(ra[0].boxes) > 0:
                boxes_a = ra[0].boxes.xywh.cpu().numpy()
                agent_centers = [get_center_xywh(b) for b in boxes_a]

        # Garbage detections
        if res.boxes is None or res.boxes.id is None or len(res.boxes) == 0:
            pass
        else:
            boxes = res.boxes.xywh.cpu().numpy()
            tids  = res.boxes.id.int().cpu().tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()

            for box, tid, det_conf in zip(boxes, tids, confs):
                cx, cy, bw, bh = map(float, box)
                raw_center = (cx, cy)

                # Resolve canonical ID (with bridging)
                if tid in alias_map:
                    canon_id = alias_map[tid]
                else:
                    adopt_id = bridge_find_orphan(raw_center, current_time)
                    if adopt_id is not None:
                        canon_id = adopt_id
                        alias_map[tid] = canon_id
                        dbg_bridges += 1
                    else:
                        canon_id = tid
                        alias_map[tid] = canon_id

                st = get_canon_state(canon_id, current_time, frame_count, raw_center)
                st['seen_this_frame'] = True

                # Initial background window
                if current_time < START_CHECK_WINDOW:
                    st['seen_in_init'] = True

                # Optional ROI refine in blurred/low-conf cases
                if ROI_ENABLE and (mode == 'object' or blur_med <= BLUR_THRESH_FALLBACK):
                    if det_conf < ROI_CONF_TRIGGER and (current_time - st['roi_last_time']) >= ROI_COOLDOWN_SEC:
                        roi_res = refine_with_roi(model_garbage, frame, (cx, cy, bw, bh),
                                                  roi_conf=ROI_CONF_TRIGGER, roi_imgsz=ROI_IMGSZ, scale=ROI_SCALE)
                        if roi_res is not None:
                            refined_center, refined_conf = roi_res
                            raw_center = refined_center
                            det_conf   = refined_conf
                            st['roi_last_time'] = current_time
                            dbg_roi_calls += 1

                # Smoothing (EMA)
                if st['last_smooth'] is None:
                    smooth = raw_center
                else:
                    sx, sy = st['last_smooth']
                    smooth = (EMA_ALPHA * raw_center[0] + (1 - EMA_ALPHA) * sx,
                              EMA_ALPHA * raw_center[1] + (1 - EMA_ALPHA) * sy)
                st['last_smooth'] = smooth
                st['positions_raw'].append(raw_center)
                st['positions_smooth'].append(smooth)
                st['last_seen_time'] = current_time
                st['last_pos'] = smooth

                # Background check once after init window
                if st['seen_in_init'] and not st['bg_checked'] and current_time >= (START_CHECK_WINDOW + 1.0 / fps):
                    st['bg_checked'] = True
                    disp_bg = calculate_displacement(st['positions_smooth'])
                    if disp_bg < move_tol_low_px:
                        st['is_background'] = True
                        dbg_bg_filtered += 1

                if st['is_background']:
                    continue

                # Update movement metrics
                disp_total = calculate_displacement(st['positions_smooth'])
                st['max_displacement'] = max(st['max_displacement'], disp_total)
                if st['max_displacement'] > min_move_px:
                    st['was_moving'] = True

                # Recent movement for stationary decision
                rmov = movement_over_recent(st['positions_smooth'], frames_window)
                is_stationary_now = (rmov < move_tol_low_px)
                is_moving_high    = (rmov > move_tol_high_px)

                # Proximity pause
                agent_near = False
                if use_proximity and agent_centers:
                    nearest = min(calculate_distance(smooth, ac) for ac in agent_centers)
                    agent_near = (nearest < proximity_thresh)

                # Age guard
                age_frames = frame_count - (st['first_seen_frame'] or frame_count)
                old_enough = (age_frames >= min_track_age_f)

                # Eligibility: was_moving OR new-appearance
                appeared_after_init = st['appeared_after_init']
                eligible_by_move = st['was_moving']
                eligible_by_new  = appeared_after_init

                eligible = (is_stationary_now and not agent_near and old_enough and (eligible_by_move or eligible_by_new))

                # Stationary state with hysteresis
                if eligible:
                    if st['stationary_start'] is None:
                        st['stationary_start'] = current_time
                    # stationary -> reset moving streak
                    st['moving_streak'] = 0

                    stationary_time = current_time - st['stationary_start']
                    if stationary_time >= stationary_dur:
                        final_timestamp = st['stationary_start']
                        if eligible_by_new and not eligible_by_move:
                            dbg_new_appearance_accepts += 1
                        elif eligible_by_move:
                            dbg_was_moving_accepts += 1
                        break
                else:
                    # Reset only after sustained "high" movement
                    if st['stationary_start'] is not None:
                        if is_moving_high:
                            st['moving_streak'] += 1
                            if st['moving_streak'] >= move_reset_frames:
                                st['stationary_start'] = None
                                st['moving_streak'] = 0
                        else:
                            st['moving_streak'] = 0

        # Orphan bookkeeping for bridging
        now_orphans = []
        for cid, st in canon_tracks.items():
            if not st.get('seen_this_frame', False):
                if (current_time - st['last_seen_time']) <= bridge_time_s:
                    now_orphans.append({'canon_id': cid, 'last_pos': st['last_pos'], 'last_time': st['last_seen_time']})
        orphans = now_orphans

    # Write output
    with open(output_txt, 'w') as f:
        if final_timestamp is not None:
            f.write(f"{final_timestamp:.2f}")

    # Write debug log
    if debug:
        dbg_lines.append("")
        dbg_lines.append(f"final_timestamp={final_timestamp:.2f}" if final_timestamp is not None else "final_timestamp=None")
        dbg_lines.append(f"bg_filtered={dbg_bg_filtered} bridges={dbg_bridges} roi_calls={dbg_roi_calls}")
        dbg_lines.append(f"accepts_new_appearance={dbg_new_appearance_accepts} accepts_was_moving={dbg_was_moving_accepts}")
        dbg_path = debug_dir / f"{video_name}.log"
        with open(dbg_path, 'w') as df:
            df.write("\n".join(dbg_lines))


# ============================================================================
#                               MAIN
# ============================================================================

def main():
    args = parse_arguments()
    input_dir = Path(args.videos)
    output_dir = Path(args.results)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(DEFAULT_GARBAGE_MODEL):
        print(f"[CRITICAL] Garbage model not found: {DEFAULT_GARBAGE_MODEL}")
        return

    print("=== IWDD Hybrid Inference ===")
    print(f"Garbage model: {DEFAULT_GARBAGE_MODEL}")
    print(f"Agent model:   {DEFAULT_AGENT_MODEL}")
    print(f"Videos:        {input_dir}")
    print(f"Results:       {output_dir}")
    print(f"CONF:          {CONF_THRESHOLD}")
    print(f"Debug logs:    {'ON' if DEBUG_ENABLED else 'OFF'} -> {output_dir / DEBUG_SUBDIR_NAME}")
    print("=============================")

    # Load models from variables (no CLI params)
    model_garbage = YOLO(DEFAULT_GARBAGE_MODEL)
    model_agents  = YOLO(DEFAULT_AGENT_MODEL) if DEFAULT_AGENT_MODEL else None

    videos = sorted(input_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos. Starting...")

    for video_file in tqdm(videos, desc="Inference", unit="vid"):
        try:
            process_single_video(str(video_file), output_dir, model_garbage, model_agents, debug=DEBUG_ENABLED)
        except Exception as e:
            with open(output_dir / f"{video_file.stem}.txt", 'w'):
                pass
            print(f"Error on {video_file.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()