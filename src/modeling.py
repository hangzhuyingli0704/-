from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAVE_XGBOOST = True
except Exception:
    _HAVE_XGBOOST = False


FEATURE_COLS = [
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

LABEL_COL = "label"


@dataclass
class TrainResult:
    model: object
    metrics: Dict[str, float]
    confusion: np.ndarray
    report: str
    feature_cols: List[str]


def load_labeled_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["source_file"] = os.path.basename(fp)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Keep only rows with labels
    if LABEL_COL not in data.columns:
        raise ValueError(f"Missing '{LABEL_COL}' column in labeled data")

    # Normalize labels: allow strings
    if data[LABEL_COL].dtype == object:
        mapping = {"low": 0, "mid": 1, "medium": 1, "high": 2, "0": 0, "1": 1, "2": 2}
        data[LABEL_COL] = data[LABEL_COL].astype(str).str.lower().map(mapping)

    data = data.dropna(subset=[LABEL_COL]).copy()
    data[LABEL_COL] = data[LABEL_COL].astype(int)

    # Ensure feature columns exist
    missing = [c for c in FEATURE_COLS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Drop rows with NaNs in features
    data = data.dropna(subset=FEATURE_COLS).copy()

    return data


def make_model(model_name: str, random_state: int = 42):
    model_name = model_name.lower().strip()

    if model_name in ("rf", "randomforest"):
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    if model_name in ("lr", "logreg", "logistic"):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto")),
            ]
        )

    if model_name in ("xgb", "xgboost"):
        if not _HAVE_XGBOOST:
            raise RuntimeError("xgboost is not installed. Try `pip install xgboost` or use --model rf.")
        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            reg_lambda=1.0,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model: {model_name}. Use rf | lr | xgb")


def train_evaluate(
    df: pd.DataFrame,
    model_name: str,
    subject_independent: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    groups = None
    if subject_independent and "subject_id" in df.columns:
        groups = df["subject_id"].astype(str).values

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        (train_idx, test_idx) = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    model = make_model(model_name, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    conf = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    rep = classification_report(y_test, y_pred, digits=4)

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return TrainResult(model=model, metrics=metrics, confusion=conf, report=rep, feature_cols=list(FEATURE_COLS))
