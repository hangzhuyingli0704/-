# MediaPipe Video-Meeting Fatigue Detector (macOS)

This repository implements a full pipeline suitable for a paper-ready project:

- Extract features from webcam frames using **MediaPipe Face Mesh** + OpenCV.
- Aggregate into fixed-length time windows (default: 30 seconds, sliding step: 5 seconds).
- Train a supervised ML classifier to output fatigue level: **LOW / MID / HIGH**.
- Run real-time inference and trigger **macOS reminders** (notification + sound), plus an on-frame overlay.
- Write logs to CSV so MATLAB can generate plots for your paper.

## 0) Recommended Python on macOS

MediaPipe typically works best with Python **3.12** (avoid 3.13).

```bash
python3.12 -m venv mp-fatigue
source mp-fatigue/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 1) Collect windowed features (optional: with labels)

### 1.1 Unlabeled session (for feature inspection)

```bash
python -m src.collect_features \
  --out data/raw_sessions/session_S01_001.csv \
  --subject_id S01 --session_id 001 \
  --show
```

Optional per-frame logging (useful for plotting EAR/MAR time series):

```bash
python -m src.collect_features \
  --out data/raw_sessions/session_S01_001.csv \
  --frame_out data/raw_sessions/session_S01_001_frames.csv \
  --subject_id S01 --session_id 001 \
  --show
```

### 1.2 Labeled session (recommended for training)

This will prompt you every emitted window to enter a label:
- 0 = LOW
- 1 = MID
- 2 = HIGH

```bash
python -m src.collect_features \
  --out data/labeled/session_S01_001_labeled.csv \
  --frame_out data/labeled/session_S01_001_frames.csv \
  --subject_id S01 --session_id 001 \
  --prompt_labels \
  --show
```

Notes:
- A 60-second **baseline calibration** runs at the start. It estimates personal EAR/MAR thresholds.
- Only CSV features are logged (no video saved).

## 2) Train a model

RandomForest (default):

```bash
python -m src.train_model \
  --data "data/labeled/*.csv" \
  --model rf \
  --out models/fatigue_model.joblib
```

Other options:
- `--model lr` (Logistic Regression)
- `--model xgb` (XGBoost, if you installed xgboost)

If you have multiple subjects and want a subject-independent split:

```bash
python -m src.train_model --data "data/labeled/*.csv" --model rf --subject_independent
```

## 3) Real-time inference + reminders

```bash
python -m src.run_realtime \
  --model models/fatigue_model.joblib \
  --subject_id S01 --session_id live \
  --out data/raw_sessions/session_S01_live_pred.csv \
  --sound "/System/Library/Sounds/Glass.aiff" \
  --show
```

Press **q** to quit.

## 4) MATLAB plots

Window-level plots:

```matlab
addpath('matlab');
plot_session('data/raw_sessions/session_S01_live_pred.csv');
```

Per-frame plots:

```matlab
addpath('matlab');
plot_frame_series('data/raw_sessions/session_S01_001_frames.csv');
```

Feature importance (if produced by training):

```matlab
addpath('matlab');
plot_feature_importance('models/fatigue_model_feature_importance.csv');
```
