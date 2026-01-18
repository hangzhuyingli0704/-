from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from .config import WindowConfig, ThresholdConfig, BaselineConfig
from .vision_features import compute_ear, compute_mar, compute_head_pose
from .aggregator import WindowAggregator
from .baseline import compute_baseline
from .io_utils import append_csv_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect MediaPipe-based fatigue features and (optionally) labels.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=str, required=True, help="Window-level CSV output path")
    p.add_argument("--frame_out", type=str, default="", help="Optional per-frame CSV output path")
    p.add_argument("--subject_id", type=str, default="S00")
    p.add_argument("--session_id", type=str, default="000")
    p.add_argument("--window", type=float, default=30.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--calib", type=float, default=60.0, help="Baseline calibration seconds")
    p.add_argument("--prompt_labels", action="store_true", help="Prompt fatigue label (0/1/2) each emitted window")
    p.add_argument("--show", action="store_true", help="Show camera window")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    wcfg = WindowConfig(window_seconds=args.window, step_seconds=args.step)
    thr = ThresholdConfig()
    bcfg = BaselineConfig(calibration_seconds=args.calib)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-3 else wcfg.default_fps

    agg = WindowAggregator(window_seconds=wcfg.window_seconds, step_seconds=wcfg.step_seconds, thr=thr, fps_hint=fps)

    # Baseline buffers
    calib_ears = []
    calib_mars = []
    baseline_set = False

    # CSV fieldnames
    base_fields = [
        "timestamp",
        "subject_id",
        "session_id",
        "ear_close_thr",
        "mar_yawn_thr",
        "label",
    ]
    feature_fields = [
        "blink_rate",
        "perclos",
        "mean_EAR",
        "EAR_var",
        "yawn_count",
        "mean_MAR",
        "MAR_peak",
        "pitch_mean",
        "pitch_var",
        "nod_count",
        "meeting_duration_s",
        "time_since_last_alert_s",
    ]
    window_fields = base_fields + feature_fields

    frame_fields = [
        "timestamp",
        "subject_id",
        "session_id",
        "EAR",
        "MAR",
        "pitch_deg",
        "yaw_deg",
        "roll_deg",
    ]

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
            yaw = None
            roll = None

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                le, re = compute_ear(lms, w, h)
                if le is not None and re is not None:
                    ear = float((le + re) / 2.0)
                mar = compute_mar(lms, w, h)
                hp = compute_head_pose(lms, w, h)
                if hp is not None:
                    pitch, yaw, roll = hp.pitch_deg, hp.yaw_deg, hp.roll_deg

            # Per-frame log (optional)
            if args.frame_out:
                append_csv_row(
                    args.frame_out,
                    frame_fields,
                    {
                        "timestamp": now,
                        "subject_id": args.subject_id,
                        "session_id": args.session_id,
                        "EAR": ear,
                        "MAR": mar,
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "roll_deg": roll,
                    },
                )

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

            # Add to window aggregator
            agg.add_frame(now, ear, mar, pitch)

            # Emit window features
            if agg.ready_to_emit(now):
                if last_alert_ts is None:
                    tsl = now - agg.session_start_ts
                else:
                    tsl = now - last_alert_ts

                feats = agg.emit_features(now, time_since_last_alert_s=tsl)

                label = ""
                if args.prompt_labels:
                    try:
                        print("\nEnter fatigue label for last window: 0=LOW, 1=MID, 2=HIGH")
                        label_in = input("label> ").strip()
                        if label_in in ("0", "1", "2"):
                            label = int(label_in)
                        else:
                            label = ""
                    except KeyboardInterrupt:
                        label = ""

                row = {
                    "timestamp": now,
                    "subject_id": args.subject_id,
                    "session_id": args.session_id,
                    "ear_close_thr": agg.ear_close_thr,
                    "mar_yawn_thr": agg.mar_yawn_thr,
                    "label": label,
                }
                row.update(feats)
                append_csv_row(args.out, window_fields, row)

                # Basic on-screen info
                if args.show:
                    msg = f"EARthr={agg.ear_close_thr:.3f} MARthr={agg.mar_yawn_thr:.3f}"
                    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"EAR={ear if ear is not None else -1:.3f} MAR={mar if mar is not None else -1:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if args.show:
                cv2.imshow("collect_features", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
