from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from .modeling import load_labeled_csvs, train_evaluate, FEATURE_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fatigue classifier from labeled window CSVs")
    p.add_argument("--data", type=str, required=True, help="Glob pattern for labeled CSVs (e.g., data/labeled/*.csv)")
    p.add_argument("--model", type=str, default="rf", choices=["rf", "lr", "xgb"])
    p.add_argument("--out", type=str, default="models/fatigue_model.joblib")
    p.add_argument("--subject_independent", action="store_true", help="Group split by subject_id if available")
    p.add_argument("--test_size", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_labeled_csvs(args.data)
    res = train_evaluate(
        df,
        model_name=args.model,
        subject_independent=args.subject_independent,
        test_size=args.test_size,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    bundle = {
        "model": res.model,
        "feature_cols": res.feature_cols,
        "label_map": {0: "low", 1: "mid", 2: "high"},
        "metrics": res.metrics,
    }
    joblib.dump(bundle, args.out)

    # Save human-readable report + confusion matrix CSV for MATLAB
    report_path = os.path.splitext(args.out)[0] + "_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("METRICS\n")
        f.write(json.dumps(res.metrics, indent=2))
        f.write("\n\nCLASSIFICATION REPORT\n")
        f.write(res.report)
        f.write("\n\nCONFUSION MATRIX (rows=true, cols=pred; order=0,1,2)\n")
        f.write(str(res.confusion))

    conf_csv = os.path.splitext(args.out)[0] + "_confusion.csv"
    conf_df = pd.DataFrame(res.confusion, index=["true0", "true1", "true2"], columns=["pred0", "pred1", "pred2"])
    conf_df.to_csv(conf_csv, index=True)

    # Optional: feature importance if model supports it
    fi_csv = os.path.splitext(args.out)[0] + "_feature_importance.csv"
    try:
        m = res.model
        if hasattr(m, "feature_importances_"):
            importances = m.feature_importances_
        elif hasattr(m, "named_steps") and "clf" in m.named_steps and hasattr(m.named_steps["clf"], "coef_"):
            # For logistic regression: use norm over classes
            importances = np.linalg.norm(m.named_steps["clf"].coef_, axis=0)
        else:
            importances = None

        if importances is not None:
            fi_df = pd.DataFrame({"feature": res.feature_cols, "importance": importances})
            fi_df.sort_values("importance", ascending=False).to_csv(fi_csv, index=False)
    except Exception:
        pass

    print("Saved model:", args.out)
    print("Saved report:", report_path)
    print("Saved confusion:", conf_csv)
    if os.path.exists(fi_csv):
        print("Saved feature importance:", fi_csv)


if __name__ == "__main__":
    main()
