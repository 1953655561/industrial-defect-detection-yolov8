#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Traditional OpenCV baseline for NEU-DET + comparison with the YOLOv8 detector.

Pipeline (all classical CV, no learning):
    gray -> CLAHE -> GaussianBlur -> defect response (Otsu-inverse / black-hat / adaptive)
         -> morphological open+close -> contour extraction -> bbox + area/aspect filtering

Parameters are tuned with a small grid search on a *train* subset (no val leakage) using
IoU>=0.3 greedy matching, then the tuned config is evaluated once on the val split.
The classical baseline cannot predict defect classes, so matching is class-agnostic;
per-class *recall* is still reported using the ground-truth class of matched boxes.

Example:
    python scripts/opencv_baseline.py --data data/neu_det --yolo-eval reports/eval_y8n_aug.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


# ------------------------------------------------------------------ io / matching utils
def load_gt(data: Path, split: str, limit: int | None = None, seed: int = 0):
    items = []
    idir, ldir = data / "images" / split, data / "labels" / split
    files = sorted(idir.glob("*.jpg"))
    if limit is not None:
        rng = random.Random(seed)
        rng.shuffle(files)
        files = files[:limit]
    for p in files:
        lf = ldir / f"{p.stem}.txt"
        boxes = []
        if lf.exists():
            for ln in lf.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                c, xc, yc, w, h = (float(v) for v in ln.split())
                img = None  # size comes from label normalization -> use 200x200 default
                boxes.append((int(c), xc, yc, w, h))
        items.append((p, boxes))
    return items


def gt_px(boxes, w, h):
    out = []
    for c, xc, yc, bw, bh in boxes:
        out.append((int(c), xc * w - bw * w / 2, yc * h - bh * h / 2,
                    xc * w + bw * w / 2, yc * h + bh * h / 2))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------ classical pipeline
def detect(gray: np.ndarray, p: dict):
    """Return list of candidate (x1,y1,x2,y2,area_ratio)."""
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (p["blur"], p["blur"]), 0)
    mode = p["mode"]
    if mode == "otsu_inv":
        _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif mode == "blackhat":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["bh"], p["bh"]))
        bh = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, k)
        bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
        _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # adaptive
        m = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, p["block"], p["C"])
    if p["close"] > 1:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close"], p["close"]))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    if p["open"] > 1:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["open"], p["open"]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        ar = area / (w * h)
        if ar < p["min_area"] or ar > p["max_area"]:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 3 or bh < 3:
            continue
        aspect = bw / max(bh, 1)
        if aspect < p["min_ar"] or aspect > p["max_ar"]:
            continue
        boxes.append((float(x), float(y), float(x + bw), float(y + bh), ar))
    return boxes


_IMG_CACHE: dict[str, np.ndarray] = {}


def read_gray(path) -> np.ndarray | None:
    key = str(path)
    if key not in _IMG_CACHE:
        img = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _IMG_CACHE[key] = img
    return _IMG_CACHE[key]


def evaluate(items, p, iou_thr=0.3):
    tp = fp = fn = 0
    per_class = {c: {"gt": 0, "hit": 0} for c in CLASSES}
    t_img = []
    for path, boxes in items:
        gray = read_gray(path)
        if gray is None:
            continue
        h, w = gray.shape
        gts = gt_px(boxes, w, h)
        t0 = time.perf_counter()
        cand = detect(gray, p)
        t_img.append((time.perf_counter() - t0) * 1000)
        for c, *_ in gts:
            per_class[CLASSES[c]]["gt"] += 1
        used = set()
        for c, *g in gts:
            best, best_iou = -1, 0.0
            for j, cb in enumerate(cand):
                if j in used:
                    continue
                v = iou(cb, g)
                if v > best_iou:
                    best, best_iou = j, v
            if best_iou >= iou_thr:
                used.add(best)
                tp += 1
                per_class[CLASSES[c]]["hit"] += 1
            else:
                fn += 1
        fp += len(cand) - len(used)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "params": p,
        "iou_threshold": iou_thr,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fp_per_image": round(fp / max(1, len(items)), 3),
        "mean_ms_per_image": round(float(np.mean(t_img)), 3) if t_img else None,
        "per_class_recall": {
            c: (round(v["hit"] / v["gt"], 4) if v["gt"] else None) for c, v in per_class.items()
        },
        "per_class_gt": {c: v["gt"] for c, v in per_class.items()},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data" / "neu_det")
    ap.add_argument("--train-subset", type=int, default=60, help="images used for grid search (train split)")
    ap.add_argument("--val-subset", type=int, default=0, help="0 = full val split")
    ap.add_argument("--yolo-eval", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "reports" / "opencv_baseline.json")
    ap.add_argument("--screens", type=Path, default=HERE / "screenshots")
    args = ap.parse_args()
    args.screens.mkdir(parents=True, exist_ok=True)

    grid = dict(
        mode=["otsu_inv", "blackhat", "adaptive"],
        clip=[2.0, 4.0],
        blur=[3, 5],
        block=[15, 25, 35],
        C=[3.0, 5.0, 10.0],
        close=[3, 5],
        open=[1, 2],
        min_area=[0.004, 0.01, 0.02],
        max_area=[0.9],
        min_ar=[0.05],
        max_ar=[20.0],
        bh=[9, 15],
    )

    train_items = load_gt(args.data, "train", args.train_subset, seed=7)
    print(f"[i] grid-search on {len(train_items)} train images")
    best = None
    tried = 0
    for mode, clip, blur, close, open_, min_area in itertools.product(
        grid["mode"], grid["clip"], grid["blur"], grid["close"], grid["open"], grid["min_area"]
    ):
        if mode == "adaptive":
            for block, C in itertools.product(grid["block"], grid["C"]):
                p = dict(mode=mode, clip=clip, blur=blur, block=block, C=C, close=close,
                         open=open_, min_area=min_area, max_area=0.9, min_ar=0.05,
                         max_ar=20.0, bh=15)
                tried += 1
                m = evaluate(train_items, p)
                if best is None or m["f1"] > best["f1"]:
                    best = m
        else:
            for bh in grid["bh"]:
                p = dict(mode=mode, clip=clip, blur=blur, block=25, C=5.0, close=close,
                         open=open_, min_area=min_area, max_area=0.9, min_ar=0.05,
                         max_ar=20.0, bh=bh)
                tried += 1
                m = evaluate(train_items, p)
                if best is None or m["f1"] > best["f1"]:
                    best = m
    print(f"[i] tried {tried} configs; best train F1={best['f1']} P={best['precision']} R={best['recall']}")

    val_items = load_gt(args.data, "val", args.val_subset or None, seed=11)
    tuned = dict(best["params"])
    final = evaluate(val_items, tuned)
    final["grid_search"] = {
        "configs_tried": tried,
        "train_images": len(train_items),
        "best_train_f1": best["f1"],
        "best_train_params": best["params"],
    }
    final["val_images"] = len(val_items)
    final["note"] = ("Traditional CV has no class head: matching is class-agnostic at IoU>=0.3; "
                     "per_class_recall uses the GT class of matched candidates. Tuned on train subset only.")

    if args.yolo_eval and args.yolo_eval.exists():
        y = json.loads(args.yolo_eval.read_text(encoding="utf-8"))
        final["yolo_comparison"] = {
            "yolo_overall": y.get("overall"),
            "yolo_per_class": y.get("per_class"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- qualitative figure: original+GT / CLAHE+mask / candidates ----
    picks = []
    for cls in CLASSES:
        for pth, boxes in val_items:
            if pth.stem.startswith(cls) and len(picks) < 6:
                picks.append((pth, boxes))
                break
    tiles = []
    for pth, boxes in picks:
        gray = cv2.imread(str(pth), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        clahe = cv2.createCLAHE(clipLimit=tuned["clip"], tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (tuned["blur"], tuned["blur"]), 0)
        if tuned["mode"] == "otsu_inv":
            _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif tuned["mode"] == "blackhat":
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tuned["bh"], tuned["bh"]))
            bh = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, k)
            bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
            _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            m = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, tuned["block"], tuned["C"])
        a = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for c, x1, y1, x2, y2 in gt_px(boxes, w, h):
            cv2.rectangle(a, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        bb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2, _ in detect(gray, tuned):
            cv2.rectangle(bb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        mask3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        tile = np.hstack([a, bb, mask3])
        tile = cv2.resize(tile, (600, 200))
        cv2.putText(tile, pth.stem.split("_")[0], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(tile, pth.stem.split("_")[0], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(tile)
    if tiles:
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - len(tiles) % 2, 2)]
        cv2.imwrite(str(args.screens / "opencv_baseline_compare.jpg"), np.vstack(rows))

    print(json.dumps({k: final[k] for k in ("precision", "recall", "f1", "fp_per_image", "mean_ms_per_image")}, indent=2))
    print(f"[OK] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
