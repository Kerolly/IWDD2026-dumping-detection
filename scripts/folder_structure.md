# IWDD Project 2026: Waste Disposal Detection
---

### 📂 `scripts/`
**Purpose:** Contains one-time scripts used for dataset preparation.

- **`01_prepare_dataset.py`** — Generates the `train_split.json` and `test_split.json` files. Run once at the start of the project.
- **`02_extract_frames.py`** — Extracts frames (`.jpg` images) from the positive clips listed in `train_split.json`, preparing them for annotation.

