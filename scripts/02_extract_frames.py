# 02_extract_frames.py

import os
import json
import cv2  # OpenCV
from pathlib import Path

# --- Main Configuration ---

# Define the time points (in seconds) around the timestamp where we want to extract frames
# It will be 10 frames: 2 before, 3 during, and 5 after the event
FRAME_OFFSETS_SECONDS = [
    -2.0, -1.0,           # Before
    -0.5, 0.0, 0.5,     # During the event
    1.0, 2.0, 3.0, 4.0, 5.0  # After
]

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data" / "splits"
VIDEO_DIR = ROOT_DIR / "data" / "videos"

# Output directory for the images we will annotate
OUTPUT_IMAGE_DIR = ROOT_DIR / "data" / "yolo_dataset" / "images"


# --- Main Function ---
def extract_frames_for_annotation():
    """
    From 'train_split.json', finds positive clips (with dumping),
    and extracts 10 frames from around the event timestamp
    Saves these frames as .jpg in 'data/yolo_dataset/images/'
    """

    print("--- Starting Frame Extraction for Annotation ---")
    
    # Ensure the output directory exists
    try:
        OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Extracted images will be saved to: {OUTPUT_IMAGE_DIR}")

    except Exception as e:
        print(f"ERROR: Could not create output directory: {e}")
        return

    # Load the list of training clips
    train_split_path = SPLITS_DIR / "train_split.json"

    if not train_split_path.exists():
        print(f"ERROR: 'train_split.json' not found in {SPLITS_DIR}")
        print("Please run the '01_prepare_dataset.py' script first.")
        return
        
    with open(train_split_path, 'r', encoding='utf-8') as f:
        train_list = json.load(f)

    # Loop through the list and process the clips
    total_videos = len(train_list)
    positive_videos = 0
    extracted_frames_count = 0

    for i, video_info in enumerate(train_list):

        # Print a progress bar
        print(f"\rProcessing video {i+1}/{total_videos}...", end="")

        # Process only positive clips with a valid timestamp
        if video_info.get("dumping") == 1 and video_info.get("timestamp") is not None:
            positive_videos += 1
            video_id = video_info["id"]
            timestamp = video_info["timestamp"]
            fps = video_info["fps"]
            duration = video_info["duration_s"]
            
            video_path = VIDEO_DIR / video_id
            
            if not video_path.exists():
                print(f"\nWarning: {video_id} - Video file not found. Skipping...")
                continue
            
            try:
                cap = cv2.VideoCapture(str(video_path)) # open the video file

                if not cap.isOpened():
                    print(f"\nWarning: {video_id} - Video file is corrupt. Skipping...")
                    continue
                
                # Calculate and extract the target frames
                for offset in FRAME_OFFSETS_SECONDS:
                    target_time_sec = timestamp + offset
                    
                    # Check if the target time is valid
                    if 0 <= target_time_sec < duration:
                        # Convert the time into a frame number
                        target_frame_num = int(target_time_sec * fps)
                        
                        # Set the video to that specific frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                        
                        ret, frame = cap.read() # read the frame
                        
                        if ret:

                            # Create a unique filename (ex: vid0001_frame_120.jpg)
                            frame_filename = f"{video_id.replace('.mp4', '')}_frame_{target_frame_num}.jpg"
                            output_path = OUTPUT_IMAGE_DIR / frame_filename
                            
                            # Save the frame as a .jpg image
                            cv2.imwrite(str(output_path), frame)
                            extracted_frames_count += 1
                
                cap.release()

            except Exception as e:
                print(f"\nERROR: processing {video_id}: {e}")

    print(f"\n--- Processing Finished ---")
    print(f"Positive clips found in training set: {positive_videos}")
    print(f"Total frames extracted (.jpg images): {extracted_frames_count}")
    print(f"Images are ready for annotation in: {OUTPUT_IMAGE_DIR}")



if __name__ == "__main__":
    extract_frames_for_annotation()