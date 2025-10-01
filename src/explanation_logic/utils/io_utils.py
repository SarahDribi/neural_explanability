# src/napx/io_utils.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv, json, logging, torch
from typing import Dict, List


log = logging.getLogger(__name__)

def make_run_dir(base: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = Path(base) / f"{ts}_{tag}" if tag else Path(base) / ts
    run.mkdir(parents=True, exist_ok=False)
    return run

def class_dir(run_dir: Path, dataset: str, image_idx: int) -> Path:
    d = run_dir / f"{dataset}_image_{image_idx}_explanations"
    d.mkdir(parents=True, exist_ok=True)
    return d

def export_csv(rows: List[Dict], path: Path) -> Path:
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    log.info("CSV written: %s", path); return path

def export_json(obj, path: Path) -> Path:
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, indent=2)
    log.info("JSON written: %s", path); return path

def export_tensor(tensor, path: Path) -> Path:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    torch.save(tensor, path); log.info("Tensor saved: %s", path); return path
