#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a trained YOLOv8 checkpoint on the val split: overall + per-class metrics.

Writes reports/eval_<name>.json, copies PR curve / confusion matrix / val batch plots
into reports/ and saves per-class AP bar chart.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent.parent


def per_class_metrics(box) -> dict:
    """Return {class_name: {p, r, ap50, ap}} defensively across ultralytics versions."""
    out = {}
    try:
        names = box.names if hasattr(box, "names") else {}
        idxs = list(getattr(box, "ap_class_index", []))
        maps = list(getattr(box, "maps", []))
        for i, ci in enumerate(idxs):
            p = r = ap50 = ap = None
            try:
                res = box.class_result(i)  # (p, r, ap50, ap) in 8.x
                if len(res) >= 4:
                    p, r, ap50, ap = res[0], res[1], res[2], res[3]
            except Exception:
                pass
            if ap50 is None:
                try:
                    ap50 = float(box.ap50[i])
                    ap = float(maps[i])
                    p = float(box.p[i])
                    r = float(box.r[i])
                except Exception:
                    ap50 = float(maps[i]) if i < len(maps) else None
            out[str(names.get(ci, ci))] = {
                "precision": float(p) if p is not None else None,
                "recall": float(r) if r is not None else None,
                "mAP50": float(ap50) if ap50 is not None else None,
                "mAP50_95": float(ap) if ap is not None else None,
            }
    except Exception as e:  # pragma: no cover
        out["_error"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "neu_det.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or args.weights.parent.parent.name
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split="val",
        plots=True,
        save_json=False,
        verbose=True,
        project=str(HERE / "runs" / "val"),
        name=name,
        exist_ok=True,
    )
    box = metrics.box
    result = {
        "weights": str(args.weights),
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        },
        "per_class": per_class_metrics(box),
        "speed_ms_per_image": {k: float(v) for k, v in dict(getattr(metrics, "speed", {})).items()},
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
    }
    reports = HERE / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"eval_{name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # copy plots
    save_dir = Path(metrics.save_dir)
    for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png",
                  "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png",
                  "val_batch0_pred.jpg"):
        src = save_dir / fname
        if src.exists():
            shutil.copy2(src, reports / f"{name}_{fname}")

    # per-class AP bar chart
    pc = {k: v for k, v in result["per_class"].items() if v.get("mAP50") is not None}
    if pc:
        names = list(pc.keys())
        ap50 = [pc[k]["mAP50"] * 100 for k in names]
        ap5095 = [(pc[k]["mAP50_95"] or 0) * 100 for k in names]
        w, h = 1000, 420
        canvas = np.full((h, w, 3), 255, np.uint8)
        left, bottom, top = 70, 360, 30
        n = len(names)
        bw = int((w - left - 30) / n / 2.2)
        for i, nm in enumerate(names):
            x = left + int(i * (w - left - 30) / n) + 10
            h1 = int(ap50[i] / 100 * (bottom - top))
            h2 = int(ap5095[i] / 100 * (bottom - top))
            cv2.rectangle(canvas, (x, bottom - h1), (x + bw, bottom), (180, 100, 40), -1)
            cv2.rectangle(canvas, (x + bw + 3, bottom - h2), (x + 2 * bw + 3, bottom), (120, 160, 200), -1)
            cv2.putText(canvas, f"{ap50[i]:.1f}", (x, bottom - h1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
            cv2.putText(canvas, nm[:12], (x, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.line(canvas, (left - 10, bottom), (w - 10, bottom), (0, 0, 0), 1)
        cv2.putText(canvas, "mAP@0.5 (dark) / mAP@0.5:0.95 (light) per class (%)", (left, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        cv2.imwrite(str(reports / f"per_class_ap_{name}.png"), canvas)

    print(json.dumps(result["overall"], indent=2))
    print(f"[OK] report -> {reports / f'eval_{name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
