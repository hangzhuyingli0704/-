from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from .config import ThresholdConfig


@dataclass
class Baseline:
    ear_mean: float
    ear_std: float
    mar_mean: float
    mar_std: float
    ear_close_threshold: float
    mar_yawn_threshold: float


class EventCounters:
    """Stateful event counters for blink/yawn/nod within a rolling time window."""

    def __init__(self, thr: ThresholdConfig):
        self.thr = thr
        # Blink state
        self._ear_below_frames = 0
        self.blink_events: Deque[float] = deque()

        # Yawn state
        self._mar_above_frames = 0
        self.yawn_events: Deque[float] = deque()

        # Nod state (pitch)
        self._nod_down = False
        self.nod_events: Deque[float] = deque()

    def reset(self) -> None:
        self._ear_below_frames = 0
        self.blink_events.clear()
        self._mar_above_frames = 0
        self.yawn_events.clear()
        self._nod_down = False
        self.nod_events.clear()

    def _trim(self, cutoff_ts: float) -> None:
        for dq in (self.blink_events, self.yawn_events, self.nod_events):
            while dq and dq[0] < cutoff_ts:
                dq.popleft()

    def update(
        self,
        ts: float,
        cutoff_ts: float,
        ear: Optional[float],
        mar: Optional[float],
        pitch_deg: Optional[float],
        ear_close_thr: float,
        mar_yawn_thr: float,
    ) -> None:
        # Blink: count an event when we exit a closed-eye run long enough
        if ear is not None and ear < ear_close_thr:
            self._ear_below_frames += 1
        else:
            if self._ear_below_frames >= self.thr.blink_min_frames:
                self.blink_events.append(ts)
            self._ear_below_frames = 0

        # Yawn: count an event when we exit a high-MAR run long enough
        if mar is not None and mar > mar_yawn_thr:
            self._mar_above_frames += 1
        else:
            if self._mar_above_frames >= self.thr.yawn_min_frames:
                self.yawn_events.append(ts)
            self._mar_above_frames = 0

        # Nod: down then up
        if pitch_deg is not None:
            if not self._nod_down:
                if pitch_deg > self.thr.nod_down_threshold_deg:
                    self._nod_down = True
            else:
                if pitch_deg < self.thr.nod_up_threshold_deg:
                    self.nod_events.append(ts)
                    self._nod_down = False

        self._trim(cutoff_ts)


class WindowAggregator:
    """Accumulates per-frame signals, outputs feature vectors on window boundaries."""

    def __init__(self, window_seconds: float, step_seconds: float, thr: ThresholdConfig, fps_hint: float = 30.0):
        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds)
        self.thr = thr
        self.fps_hint = float(fps_hint)

        self.ears: Deque[float] = deque()
        self.mars: Deque[float] = deque()
        self.pitches: Deque[float] = deque()

        self._ts: Deque[float] = deque()
        self._event = EventCounters(thr)

        self.session_start_ts = time.time()
        self.last_emit_ts: Optional[float] = None

        # Baseline thresholds (can be set after calibration)
        self.ear_close_thr = thr.ear_close_threshold
        self.mar_yawn_thr = thr.mar_yawn_threshold

    def set_thresholds(self, ear_close_thr: float, mar_yawn_thr: float) -> None:
        self.ear_close_thr = float(ear_close_thr)
        self.mar_yawn_thr = float(mar_yawn_thr)

    def add_frame(self, timestamp: float, ear: Optional[float], mar: Optional[float], pitch_deg: Optional[float]) -> None:
        self._ts.append(timestamp)

        self.ears.append(float(ear) if ear is not None else np.nan)
        self.mars.append(float(mar) if mar is not None else np.nan)
        self.pitches.append(float(pitch_deg) if pitch_deg is not None else np.nan)

        cutoff = timestamp - self.window_seconds
        self._event.update(timestamp, cutoff, ear, mar, pitch_deg, self.ear_close_thr, self.mar_yawn_thr)

        self._trim_old(timestamp)

    def _trim_old(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_seconds
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
            self.ears.popleft()
            self.mars.popleft()
            self.pitches.popleft()

    def ready_to_emit(self, now_ts: float) -> bool:
        if not self._ts:
            return False
        if self._ts[-1] - self._ts[0] < self.window_seconds * 0.8:
            return False
        if self.last_emit_ts is None:
            return True
        return (now_ts - self.last_emit_ts) >= self.step_seconds

    def emit_features(self, now_ts: float, time_since_last_alert_s: Optional[float] = None) -> Dict[str, float]:
        ears = np.array(self.ears, dtype=float)
        mars = np.array(self.mars, dtype=float)
        pitches = np.array(self.pitches, dtype=float)

        ears_valid = ears[~np.isnan(ears)]
        mars_valid = mars[~np.isnan(mars)]
        pitch_valid = pitches[~np.isnan(pitches)]

        mean_ear = float(np.mean(ears_valid)) if ears_valid.size else 0.0
        var_ear = float(np.var(ears_valid)) if ears_valid.size else 0.0
        perclos = float(np.mean(ears_valid < self.ear_close_thr)) if ears_valid.size else 0.0

        window_len = float(self._ts[-1] - self._ts[0]) if len(self._ts) >= 2 else self.window_seconds
        if window_len <= 0:
            window_len = self.window_seconds

        blink_rate = float(len(self._event.blink_events)) * 60.0 / window_len

        mean_mar = float(np.mean(mars_valid)) if mars_valid.size else 0.0
        mar_peak = float(np.max(mars_valid)) if mars_valid.size else 0.0
        yawn_count = float(len(self._event.yawn_events))

        pitch_mean = float(np.mean(pitch_valid)) if pitch_valid.size else 0.0
        pitch_var = float(np.var(pitch_valid)) if pitch_valid.size else 0.0
        nod_count = float(len(self._event.nod_events))

        meeting_duration_s = float(now_ts - self.session_start_ts)
        tsl = float(time_since_last_alert_s) if time_since_last_alert_s is not None else -1.0

        self.last_emit_ts = now_ts

        return {
            "blink_rate": blink_rate,
            "perclos": perclos,
            "mean_EAR": mean_ear,
            "EAR_var": var_ear,
            "yawn_count": yawn_count,
            "mean_MAR": mean_mar,
            "MAR_peak": mar_peak,
            "pitch_mean": pitch_mean,
            "pitch_var": pitch_var,
            "nod_count": nod_count,
            "meeting_duration_s": meeting_duration_s,
            "time_since_last_alert_s": tsl,
        }
