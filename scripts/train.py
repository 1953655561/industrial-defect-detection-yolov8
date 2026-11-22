#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLOv8 training wrapper with reproducible settings and a results summary.

Examples:
    python scripts/train.py --model yolov8n.pt --epochs 100 --name y8n_aug
    python scripts/train.py --model yolov8n.pt --epochs 100 --name y8n_noaug --aug none
    python scripts/train.py --model yolov8s.pt --epochs 100 --name y8s_aug
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent.parent

NO_AUG = dict(
    mosaic=0.0,
    close_mosaic=0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    bgr=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    auto_augment=None,
)


def summarize_csv(results_csv: Path) -> dict:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    out = {"epochs_logged": int(len(df))}
    if "metrics/mAP50(B)" in df.columns:
        best = df.loc[df["metrics/mAP50(B)"].idxmax()]
        out.update(
            {
                "best_epoch": int(best["epoch"]),
                "best_mAP50": float(best["metrics/mAP50(B)"]),
                "best_mAP50_95": float(best["metrics/mAP50-95(B)"]),
                "best_precision": float(best["metrics/precision(B)"]),
                "best_recall": float(best["metrics/recall(B)"]),
                "final_mAP50": float(df["metrics/mAP50(B)"].iloc[-1]),
            }
        )
    if "time" in df.columns:
        out["train_seconds_logged"] = float(df["time"].iloc[-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--data", type=Path, default=HERE / "data" / "neu_det.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers; 0 is the safe default on Windows with limited RAM, "
                         "spawned workers are not always joined between val passes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--aug", choices=["default", "none"], default="default")
    ap.add_argument("--cache", choices=["none", "ram", "disk"], default="none",
                    help="dataset caching; 'none' keeps RAM usage low on 16 GB machines")
    args = ap.parse_args()

    runs = HERE / "runs" / "detect"
    runs.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(runs),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
        patience=args.patience,
        cache=False if args.cache == "none" else args.cache,
        verbose=True,
        amp=True,
    )
    if args.aug == "none":
        kwargs.update(NO_AUG)

    args_meta = dict(args=dict(vars(args)), resolved=str(runs / args.name), aug=args.aug)
    (runs / args.name).mkdir(parents=True, exist_ok=True)
    (runs / args.name / "run_meta.json").write_text(
        json.dumps(args_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    t0 = time.time()
    model.train(**kwargs)
    elapsed = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    results_csv = save_dir / "results.csv"
    summary = {"name": args.name, "model": args.model, "wall_seconds": round(elapsed, 1)}
    if results_csv.exists():
        summary.update(summarize_csv(results_csv))
    reports = HERE / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"train_{args.name}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
