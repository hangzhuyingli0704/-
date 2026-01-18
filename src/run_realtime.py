from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import joblib

from .config import WindowConfig, ThresholdConfig, BaselineConfig
from .vision_features import compute_ear, compute_mar, compute_head_pose
from .aggregator import WindowAggregator
from .baseline import compute_baseline
from .macos_alert import alert
from .io_utils import append_csv_row
from .modeling import FEATURE_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time fatigue inference with reminders on macOS")
    p.add_argument("--model", type=str, required=True, help="Path to trained model bundle (*.joblib)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=str, default="data/raw_sessions/realtime_log.csv", help="Window-level log CSV")
    p.add_argument("--window", type=float, default=30.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--calib", type=float, default=60.0)
    p.add_argument("--subject_id", type=str, default="S00")
    p.add_argument("--session_id", type=str, default="realtime")
    p.add_argument("--sound", type=str, default="/System/Library/Sounds/Glass.aiff")
    p.add_argument("--prob_thr", type=float, default=0.55, help="Min probability for HIGH class to trigger")
    p.add_argument("--smooth_k", type=int, default=3, help="Majority vote over last k windows")
    p.add_argument("--cooldown", type=float, default=90.0, help="Cooldown seconds between alerts")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _predict(bundle, feat_dict) -> Tuple[int, float]:
    model = bundle["model"]
    cols = bundle["feature_cols"]
    x = np.array([[feat_dict.get(c, 0.0) for c in cols]], dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        p_high = float(proba[2])
    else:
        pred = int(model.predict(x)[0])
        p_high = 0.0
    return pred, p_high


def main() -> None:
    args = parse_args()

    bundle = joblib.load(args.model)

    wcfg = WindowConfig(window_seconds=args.window, step_seconds=args.step)
    thr = ThresholdConfig(alert_cooldown_seconds=args.cooldown)
    bcfg = BaselineConfig(calibration_seconds=args.calib)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-3 else wcfg.default_fps

    agg = WindowAggregator(window_seconds=wcfg.window_seconds, step_seconds=wcfg.step_seconds, thr=thr, fps_hint=fps)

    calib_ears = []
    calib_mars = []
    baseline_set = False

    # Logging
    base_fields = [
        "timestamp",
        "subject_id",
        "session_id",
        "pred",
        "p_high",
        "ear_close_thr",
        "mar_yawn_thr",
    ]
    log_fields = base_fields + FEATURE_COLS

    # Smoothing
    recent_preds = deque(maxlen=max(1, int(args.smooth_k)))

    last_alert_ts: Optional[float] = None

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(frame_rgb)

            h, w = frame.shape[:2]
            ear = None
            mar = None
            pitch = None

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                le, re = compute_ear(lms, w, h)
                if le is not None and re is not None:
                    ear = float((le + re) / 2.0)
                mar = compute_mar(lms, w, h)
                hp = compute_head_pose(lms, w, h)
                if hp is not None:
                    pitch = hp.pitch_deg

            # Baseline calibration
            if not baseline_set and (now - t0) <= bcfg.calibration_seconds:
                if ear is not None and mar is not None:
                    calib_ears.append(ear)
                    calib_mars.append(mar)
            elif not baseline_set:
                bl = compute_baseline(calib_ears, calib_mars, bcfg)
                if bl is not None:
                    agg.set_thresholds(bl.ear_close_thr, bl.mar_yawn_thr)
                    baseline_set = True

            agg.add_frame(now, ear, mar, pitch)

            pred = None
            p_high = 0.0
            smoothed = None

            if agg.ready_to_emit(now):
                if last_alert_ts is None:
                    tsl = now - agg.session_start_ts
                else:
                    tsl = now - last_alert_ts

                feats = agg.emit_features(now, time_since_last_alert_s=tsl)

                pred, p_high = _predict(bundle, feats)
                recent_preds.append(pred)

                # Majority vote smoothing
                counts = {0: 0, 1: 0, 2: 0}
                for p in recent_preds:
                    counts[p] += 1
                smoothed = max(counts.items(), key=lambda kv: kv[1])[0]

                # Log row
                row = {
                    "timestamp": now,
                    "subject_id": args.subject_id,
                    "session_id": args.session_id,
                    "pred": int(smoothed),
                    "p_high": float(p_high),
                    "ear_close_thr": float(agg.ear_close_thr),
                    "mar_yawn_thr": float(agg.mar_yawn_thr),
                }
                row.update({c: feats.get(c, 0.0) for c in FEATURE_COLS})
                append_csv_row(args.out, log_fields, row)

                # Alert logic: need HIGH after smoothing + prob threshold + cooldown
                want_alert = (smoothed == 2) and (p_high >= args.prob_thr)
                cooldown_ok = (last_alert_ts is None) or ((now - last_alert_ts) >= thr.alert_cooldown_seconds)

                if want_alert and cooldown_ok:
                    alert(
                        title="Fatigue Monitor",
                        message="High fatigue detected. Please take a short break (20-30s).",
                        sound_path=args.sound,
                    )
                    last_alert_ts = now

            if args.show:
                # Overlay
                status = "CALIBRATING" if not baseline_set else "RUNNING"
                cv2.putText(frame, f"{status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"EARthr={agg.ear_close_thr:.3f} MARthr={agg.mar_yawn_thr:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if pred is not None:
                    label = {0: "LOW", 1: "MID", 2: "HIGH"}.get(int(smoothed), str(smoothed))
                    cv2.putText(frame, f"PRED={label}  pHigh={p_high:.2f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("fatigue_realtime", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
