import os
import json
import cv2  # OpenCV
from pathlib import Path

# --- Main Configuration ---

# Target total number of background images
MAX_BACKGROUND_IMAGES = 700

# Number of frames to extract per negative video (increased for better coverage)
FRAMES_PER_VIDEO = 10

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data" / "splits"
VIDEO_DIR = ROOT_DIR / "data" / "videos"

# Output directory
OUTPUT_IMAGE_DIR = ROOT_DIR / "data" / "yolo_dataset" / "clean_images"

# --- Main Function ---
def extract_negative_frames():
    """
    Scans 'train_split.json' for NEGATIVE clips (dumping == 0).
    Extracts multiple evenly-spaced frames from each video until target of ~700 images is reached.
    These images serve as 'background' samples to reduce False Positives.
    """

    print("--- Starting Background Frame Extraction ---")
    print(f"Target: ~{MAX_BACKGROUND_IMAGES} images total")
    
    # Ensure the output directory exists
    try:
        OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create output directory: {e}")
        return

    # Load the training list
    train_split_path = SPLITS_DIR / "train_split.json"
    if not train_split_path.exists():
        print(f"ERROR: 'train_split.json' not found.")
        return
        
    with open(train_split_path, 'r', encoding='utf-8') as f:
        train_list = json.load(f)

    extracted_frames_count = 0
    processed_videos = 0

    for i, video_info in enumerate(train_list):
        # 1. STOP CONDITION: If we reached the target, stop processing
        if extracted_frames_count >= MAX_BACKGROUND_IMAGES:
            print(f"\n\nReached target of {extracted_frames_count} images. Stopping.")
            break

        # 2. FILTER: Process only NEGATIVE clips (dumping == 0)
        if video_info.get("dumping") == 0:
            
            video_id = video_info["id"]
            fps = video_info["fps"]
            duration = video_info["duration_s"]
            
            # Skip invalid videos
            if duration <= 0 or fps <= 0:
                continue

            video_path = VIDEO_DIR / video_id
            if not video_path.exists():
                continue
            
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue
                
                # 3. STRATEGY: Extract frames at regular intervals throughout the video
                # Generate 10 evenly-spaced time points across the video duration
                target_times = []
                if duration > 0:
                    interval = duration / (FRAMES_PER_VIDEO + 1)
                    for i in range(1, FRAMES_PER_VIDEO + 1):
                        target_times.append(interval * i)
                
                # For very short videos, take fewer frames
                if duration < 1.0:
                    target_times = [duration * 0.5]

                frames_extracted_from_this_video = 0

                for target_time in target_times:
                    # Check global limit again inside the inner loop
                    if extracted_frames_count >= MAX_BACKGROUND_IMAGES:
                        break

                    target_frame_num = int(target_time * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                    
                    ret, frame = cap.read()
                    
                    if ret:
                        # Naming convention: explicit 'neg' prefix to identify them easily
                        frame_filename = f"neg_{video_id.replace('.mp4', '')}_frame_{target_frame_num}.jpg"
                        output_path = OUTPUT_IMAGE_DIR / frame_filename
                        
                        cv2.imwrite(str(output_path), frame)
                        
                        extracted_frames_count += 1
                        frames_extracted_from_this_video += 1
                
                cap.release()
                processed_videos += 1
                print(f"\rExtracted {frames_extracted_from_this_video} frames from {video_id}. Total: {extracted_frames_count}/{MAX_BACKGROUND_IMAGES}", end="")

            except Exception as e:
                print(f"\nError processing {video_id}: {e}")

    print(f"\n\n--- Extraction Finished ---")
    print(f"Total negative videos processed: {processed_videos}")
    print(f"Total background images created: {extracted_frames_count}")
    print(f"Location: {OUTPUT_IMAGE_DIR}")

if __name__ == "__main__":
    extract_negative_frames()