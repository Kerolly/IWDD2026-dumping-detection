import json
import shutil
import os
from pathlib import Path
from tqdm import tqdm  

# --- CONFIGURATION ---
# Define project root relative to this script location
ROOT_DIR = Path(__file__).resolve().parent.parent

# Input paths
VAL_SPLIT_PATH = ROOT_DIR / "data" / "splits" / "val_split.json"
SOURCE_VIDEO_DIR = ROOT_DIR / "data" / "videos"

# Output path
DEST_VIDEO_DIR = ROOT_DIR / "data" / "val_videos"

def organize_validation_set():
    """
    Reads the validation split JSON and copies the corresponding video files
    from the main video directory to a dedicated validation folder.
    """
    print("--- Organizing Validation Dataset ---")
    
    # 1. Check if source files exist
    if not VAL_SPLIT_PATH.exists():
        print(f"[ERROR] Split file not found: {VAL_SPLIT_PATH}")
        return
    
    if not SOURCE_VIDEO_DIR.exists():
        print(f"[ERROR] Source video directory not found: {SOURCE_VIDEO_DIR}")
        return

    # 2. Create destination directory if it doesn't exist
    try:
        DEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Target directory ready: {DEST_VIDEO_DIR}")
    except Exception as e:
        print(f"[ERROR] Could not create directory: {e}")
        return

    # 3. Load validation list
    print(f"Loading list from {VAL_SPLIT_PATH.name}...")
    with open(VAL_SPLIT_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    total_videos = len(val_data)
    print(f"Found {total_videos} videos in validation set.")

    # 4. Copy files
    copied_count = 0
    missing_count = 0

    print("Starting copy process (this may take a while)...")
    
    # Use tqdm for a progress bar if available, otherwise simple loop
    iterator = tqdm(val_data, desc="Copying") if 'tqdm' in globals() else val_data

    for entry in iterator:
        video_filename = entry['id']
        
        source_path = SOURCE_VIDEO_DIR / video_filename
        dest_path = DEST_VIDEO_DIR / video_filename

        if source_path.exists():
            try:
                # shutil.copy2 preserves metadata (timestamps)
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                if 'tqdm' not in globals():
                    print(f"Copied: {video_filename}")
            except Exception as e:
                print(f"[ERROR] Failed to copy {video_filename}: {e}")
        else:
            print(f"[WARNING] Source file missing: {video_filename}")
            missing_count += 1

    # 5. Final Summary
    print("\n--- Process Complete ---")
    print(f"Successfully copied: {copied_count}")
    print(f"Missing files: {missing_count}")
    print(f"Validation videos are located in: {DEST_VIDEO_DIR}")

if __name__ == "__main__":
    organize_validation_set()