# IWDD Project 2026: Waste Disposal Detection
---

### 📂 `src/`
**Purpose:** Main source code of the solution. These are the Python modules imported and used by `test.py`.

- **`src/detector.py`** — Encapsulates the logic for running the YOLO model (loading the model, making predictions on a frame).
- **`src/tracker.py`** — Handles the integration of the tracker (e.g., DeepSORT) with YOLO detections.
- **`src/rules_engine.py`** — The "brain" of the solution. Contains the business logic (e.g., "a person dropped an object and walked away", "the object remains stationary for N frames").
- **`src/utils/`** — Helper functions, following the provided guidelines.
  - `video_io.py` — Functions for reading/writing video files (using OpenCV).
  - `postproc.py` — Post-processing functions (e.g., smoothing, selecting the final timestamp from multiple detections).
