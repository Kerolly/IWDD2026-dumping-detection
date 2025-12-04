import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# --- CONFIGURATION ---
# Directory containing prediction text files (.txt)
PREDICTIONS_DIR = 'resultsV7'

# Directory containing ground-truth annotation files (.json)
GROUND_TRUTH_DIR = 'data/annotations'

# (Optional) JSON file listing validation videos. If missing or None,
# the script will evaluate all prediction files in the predictions folder.
SPLIT_FILE = 'data/splits/val_split.json'

# --- EVALUATION RULES ---
TOLERANCE_PRE = 3.0   # Seconds before event (-3s)
TOLERANCE_POST = 10.0 # Seconds after event (+10s)

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1

def evaluate():
    pred_dir = Path(PREDICTIONS_DIR)
    gt_dir = Path(GROUND_TRUTH_DIR)
    
    print(f"--- IWDD Scoring Engine ---")
    print(f"Predictions: {pred_dir}")
    print(f"Ground Truth: {gt_dir}")

    # 1. Select files and build ground truth catalog
    target_ids = []
    
    if SPLIT_FILE and os.path.exists(SPLIT_FILE):
        print(f"Using validation split: {SPLIT_FILE}")
        with open(SPLIT_FILE, 'r') as f:
            data = json.load(f)
            target_ids = [item['id'].replace('.mp4', '') for item in data]
    else:
        print("WARNING: No split file found. Evaluating all prediction files in the predictions folder.")
        target_ids = [f.stem for f in pred_dir.glob("*.txt")]

    # Load all ground truth data to track total positives/negatives
    gt_catalog = {}
    total_positives = 0  # Total videos with dumping in ground truth
    total_negatives = 0  # Total videos without dumping in ground truth
    
    for vid_id in target_ids:
        json_path = gt_dir / f"{vid_id}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                gt_data = json.load(f)
                gt_catalog[vid_id] = gt_data
                has_dumping = gt_data.get('Dumping', 0) == 1
                if has_dumping:
                    total_positives += 1
                else:
                    total_negatives += 1
    
    print(f"Total videos in split: {len(target_ids)}")
    print(f"Total positives (with dumping): {total_positives}")
    print(f"Total negatives (no dumping): {total_negatives}")

    # 2. Counters
    tp = 0; fp = 0; fn = 0; tn = 0
    evaluated_count = 0  # Videos with both GT and prediction
    unevaluated_positives = 0  # Videos missing prediction but have dumping
    unevaluated_negatives = 0  # Videos missing prediction and no dumping

    # 3. Evaluation loop
    for vid_id in target_ids:
        # A. Load ground truth (JSON)
        if vid_id not in gt_catalog:
            print(f"Skip: Missing JSON for {vid_id}")
            continue
        
        gt_data = gt_catalog[vid_id]
        
        has_dumping = gt_data.get('Dumping', 0) == 1
        gt_timestamp = gt_data.get('DumpingDetails', {}).get('Timestamp', None)

        # B. Load prediction (TXT)
        txt_path = pred_dir / f"{vid_id}.txt"
        pred_timestamp = None
        
        if txt_path.exists():
            try:
                content = txt_path.read_text().strip()
                if content:
                    pred_timestamp = float(content)
            except ValueError:
                pass  # Empty or corrupt file
        else:
            # Prediction file missing - count as unevaluated
            if has_dumping:
                unevaluated_positives += 1
            else:
                unevaluated_negatives += 1
            print(f"MISSING: No prediction file for {vid_id}")
            continue
        
        evaluated_count += 1

        # C. Logical comparison
        if has_dumping:
            # Positive case (ground truth contains dumping)
            if pred_timestamp is not None:
                diff = pred_timestamp - gt_timestamp
                # Check time window [-3s, +10s]
                if -TOLERANCE_PRE <= diff <= TOLERANCE_POST:
                    tp += 1
                else:
                    fp += 1  # Found but at wrong time
                    fn += 1  # Also counts as missed real event
                    print(f"ERROR: {vid_id}: Time Mismatch (Real: {gt_timestamp}, Pred: {pred_timestamp})")
            else:
                fn += 1  # No prediction found (false negative)
                print(f"ERROR: {vid_id}: False Negative (Missed)")
        
        else:
            # Negative case (no dumping in ground truth)
            if pred_timestamp is not None:
                fp += 1
                print(f"ERROR: {vid_id}: False Positive (Alarm on clean video)")
            else:
                tn += 1

    # 4. Final results
    precision, recall, f1 = calculate_f1(tp, fp, fn)

    # Prepare results output
    results_lines = [
        "",
        "=" * 50,
        "FINAL RESULTS",
        "=" * 50,
        f"Folder: {PREDICTIONS_DIR}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total Videos in Dataset:     {len(target_ids)}",
        f"Evaluated Videos:            {evaluated_count}",
        f"Unevaluated (missing pred):  {unevaluated_positives + unevaluated_negatives}",
        "-" * 50,
        f"True Positives:  {tp} (correct)",
        f"False Positives: {fp} (false alarms / wrong time)",
        f"False Negatives: {fn} (missed events)",
        f"True Negatives:  {tn} (correctly ignored)",
        f"Unevaluated Positives: {unevaluated_positives} (missing predictions)",
        f"Unevaluated Negatives: {unevaluated_negatives} (missing predictions)",
        "-" * 50,
        f"PRECISION:  {precision:.4f}",
        f"RECALL:     {recall:.4f}",
        f"F1-SCORE:   {f1:.4f}",
        "-" * 50,
        f"METRICS BASED ON: {evaluated_count}/{len(target_ids)} evaluated videos",
        "=" * 50,
    ]

    # Print to console
    for line in results_lines:
        print(line)
    
    # Save results to file in root evaluation folder
    results_dir = Path(__file__).parent.parent / 'evaluation'
    results_dir.mkdir(exist_ok=True)
    
    folder_name = Path(PREDICTIONS_DIR).name

    results_file = results_dir / f"scores_{folder_name}.txt"
    
    with open(results_file, 'w') as f:
        f.write('\n'.join(results_lines))
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    evaluate()