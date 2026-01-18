from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_csv_header(path: str, fieldnames: List[str]) -> None:
    ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_csv_row(path: str, fieldnames: List[str], row: Dict) -> None:
    # If file doesn't exist, create with header
    if not os.path.exists(path):
        write_csv_header(path, fieldnames)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: row.get(k, "") for k in fieldnames})
