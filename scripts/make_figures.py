#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate README figures: prediction-vs-ground-truth grids and dataset statistics.

    python scripts/make_figures.py --weights runs/detect/y8n_aug/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent.parent
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def load_gt(lf: Path, w: int, h: int):
    out = []
    if not lf.exists():
        return out
    for ln in lf.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        c, xc, yc, bw, bh = (float(v) for v in ln.split())
        out.append((int(c), (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def draw(img, boxes, color, cls_first=True):
    out = img.copy()
    for c, x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        if cls_first:
            label = CLASSES[int(c)] if 0 <= int(c) < len(CLASSES) else str(int(c))
            cv2.putText(out, label, (int(x1) + 1, max(int(y1) - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, color, 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HERE / "runs" / "detect" / "y8n_aug" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=HERE / "data" / "neu_det")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    screens = HERE / "screenshots"
    screens.mkdir(exist_ok=True)

    # ---------- class statistics ----------
    analysis = json.loads((HERE / "reports" / "dataset_analysis.json").read_text(encoding="utf-8"))
    inst = analysis["instances_per_class"]
    names = list(inst.keys())
    canvas = np.full((360, 900, 3), 255, np.uint8)
    mx = max(inst.values()) if inst else 1
    bw = 90
    for i, nm in enumerate(names):
        x = 70 + i * 135
        hgt = int(inst[nm] / mx * 240)
        cv2.rectangle(canvas, (x, 300 - hgt), (x + bw, 300), (170, 110, 60), -1)
        cv2.putText(canvas, str(inst[nm]), (x + 10, 300 - hgt - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(canvas, nm[:11], (x - 4, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(canvas, (50, 300), (880, 300), (0, 0, 0), 1)
    cv2.putText(canvas, "NEU-DET instances per class (total %d)" % sum(inst.values()),
                (70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.imwrite(str(screens / "class_distribution.png"), canvas)

    # ---------- prediction vs GT ----------
    vdir, ldir = args.data / "images" / "val", args.data / "labels" / "val"
    files = sorted(vdir.glob("*.jpg"))
    rng = random.Random(args.seed)
    picks = []
    for cls in CLASSES:
        cand = [p for p in files if p.stem.startswith(cls)]
        if cand:
            picks.append(rng.choice(cand))
    picks = picks[: args.n] or rng.sample(files, min(args.n, len(files)))
    model = YOLO(str(args.weights)) if args.weights.exists() else None

    tiles = []
    for p in picks:
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        gt = load_gt(ldir / f"{p.stem}.txt", w, h)
        left = draw(img, gt, (0, 220, 0))
        cv2.putText(left, "GT", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)
        if model is not None:
            r = model.predict(source=str(p), imgsz=640, conf=args.conf, verbose=False)[0]
            preds = [(int(c), *b) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]
            right = draw(img, preds, (0, 0, 230))
            cv2.putText(right, "YOLOv8", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 230), 1)
        else:
            right = img.copy()
        tiles.append(np.hstack([left, right]))
    tiles = [cv2.resize(t, (520, 260)) for t in tiles]
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(screens / "pred_vs_gt.jpg"), np.vstack(rows))
    print(f"[OK] {screens / 'pred_vs_gt.jpg'}")
    print(f"[OK] {screens / 'class_distribution.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
