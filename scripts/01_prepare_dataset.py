# 01_prepare_dataset.py

import os 
import json
import random
from pathlib import Path
import cv2
import math

# Set random seed for reproducibility
RANDOM_SEED = 42

# Split ratios
SPLIT_RATIO = 0.8 # 80% train, 20% validation

# Tolerance for frame extraction and duration (seconds) for metadata validation
DURATION_TOLERANCE = 0.5 # 0.5 seconds tolerance
FPS_TOLERANCE = 1 # 1 fps tolerance

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent
#print(f"Root directory: {ROOT_DIR}")
ANNOTATION_DIR = ROOT_DIR / "data" / "annotations"
VIDEO_DIR = ROOT_DIR / "data" / "videos"
SPLITS_DIR = ROOT_DIR / "data" / "splits"


# --- Functions ---

def validate_video_file(video_path):
    """
    Check if the video file can be opened and read.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        cv2.VideoCapture: Video capture object if valid
        None: otherwise
    """

    if not video_path.exists():
        print(f"Video file does not exist: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return None
    
    # If opened successfully, return the capture object
    return cap


def validate_metadata(json_info, cap, video_id):
    """
    Validate video metadata against JSON info.

    Args:
        json_info (dict): Metadata from JSON file.
        cap (cv2.VideoCapture): Video capture object.
        video_id (str): ID of the video for logging.

    Returns:
        dict: With real metadata, if is valid
        None: otherwise
    """

    # Extract metadata from JSON
    json_duration = float(json_info.get("Duration (s)"))
    json_fps = float(json_info.get("Frame Rate (fps)"))

    #print(f"Duration from json: {json_duration}\nFps from json: {json_fps}")

    # Extract metadata from video
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    real_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Safety check for zero fps
    if real_fps == 0 or real_frame_count == 0:
        print(f"Invalid video metadata (zero fps or frame count) for video ID: {video_id}")
        return None
    
    real_duration = real_frame_count / real_fps

    # Check the data
    duration_diff = abs(json_duration - real_duration)
    fps_diff = abs(json_fps - real_fps)

    if duration_diff > DURATION_TOLERANCE:
        print(f"Duration mismatch for video ID: {video_id}. JSON: {json_duration}, Real: {real_duration}")
        return None
    
    if fps_diff > FPS_TOLERANCE:
        print(f"FPS mismatch for video ID: {video_id}. JSON: {json_fps}, Real: {real_fps}")
        return None
    
    return {
        "duration_s": real_duration,
        "fps": real_fps
        }



def process_dataset():
    """
    Principal function
    
    1. Find all .json files.
    2. Validate them (JSON + Video + Metadata).
    3. Create a list of valid data.
    4. Shuffle and split the list.
    5. Save 'train_split.json' and 'val_split.json' files.
    """

    print(f"--- Begin the validation process  ---")
    print(f"Annotations dir: {ANNOTATION_DIR}")
    print(f"Videos dir: {VIDEO_DIR}\n")

    all_valid_videos = []
    total_files = 0
    invalid_files = 0

    annotation_files = list(ANNOTATION_DIR.glob("*.json")) # get all .json files
    total_files = len(annotation_files)
    
    if total_files == 0:
        print(f"Error: There is no .json files {ANNOTATION_DIR}")
        return

    print(f"Found {total_files} annotation files\nBeginning validation...\n")

    # Check each .json file
    for json_file in annotation_files:
        try:
            # 1. JSON validation
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_id = data.get("ID")
            json_info = data.get("Video Info")

            if not video_id or not json_info:
                print(f"Warning: {json_file.name}: Invalid JSON structure.")
                invalid_files += 1
                continue

            # 2. Video file validation
            video_path = VIDEO_DIR / video_id
            cap = validate_video_file(video_path)

            if cap is None:
                invalid_files += 1
                continue
            
            # 3. Metadata validation
            real_video_stats = validate_metadata(json_info, cap, video_id)
            cap.release()  # Release the video capture object
            
            if real_video_stats is None:
                invalid_files += 1
                continue
            
            # --- Success, this video is valid ---
            dumping_status = data.get("Dumping")
            timestamp = None
            time_of_dumping = None

            if dumping_status == 1:
                timestamp = data.get("DumpingDetails", {}).get("Timestamp")
                time_of_dumping = data.get("DumpingDetails", {}).get("Time of Dumping")
                
            all_valid_videos.append({
                "id": video_id,
                "dumping": dumping_status,
                "time_of_dumping": time_of_dumping,
                "timestamp": timestamp,
                "duration_s": real_video_stats["duration_s"],
                "fps": real_video_stats["fps"]
            })

        except Exception as e:
            print(f"Error: {json_file.name}: {e}")
            invalid_files += 1

    # --- Print the results ---
    print(f"\n--- Validation results ---")
    print(f"Total processed files: {total_files}")
    print(f"Valid files: {len(all_valid_videos)}")
    print(f"Invalid files (removed): {invalid_files}")
    print(f"--------------------------\n")
    
    if not all_valid_videos:
        print("Error: No valid videos")
        return

    # 4. Shuffle and split the list
    print(f"Shuffle {len(all_valid_videos)} valid videos (Seed: {RANDOM_SEED})...")
    random.seed(RANDOM_SEED)
    random.shuffle(all_valid_videos)
    
    split_index = int(len(all_valid_videos) * SPLIT_RATIO)
    train_list = all_valid_videos[:split_index]
    val_list = all_valid_videos[split_index:]

    print(f"Data was split into:")
    print(f"  -> Train set: {len(train_list)} videos")
    print(f"  -> Validation set: {len(val_list)} videos")

    # 5. Save the files
    try:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(SPLITS_DIR / "train_split.json", 'w', encoding='utf-8') as f:
            json.dump(train_list, f, indent=4)
        
        with open(SPLITS_DIR / "val_split.json", 'w', encoding='utf-8') as f:
            json.dump(val_list, f, indent=4)
            
        print(f"\nSuccess, file was saved {SPLITS_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the dataset preparation process
    process_dataset()