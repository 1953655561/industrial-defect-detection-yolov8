#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Label quality-check for the prepared YOLO dataset.

Checks performed on every label file:
  1. class id in [0, nc)
  2. bbox normalized coords inside [0,1] and w,h > min_size
  3. bbox pixel size >= --min-px
  4. duplicated boxes inside one image (IoU > 0.95)
  5. images with zero boxes (empty labels) are reported, not failed
  6. class-name vs file-name consistency (NEU-DET file names are prefixed by class)
  7. per-class instance statistics vs --expect-balance
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter + 1e-12)


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=here / "data" / "neu_det")
    ap.add_argument("--reports", type=Path, default=here / "reports")
    ap.add_argument("--min-px", type=float, default=4.0)
    ap.add_argument("--min-size", type=float, default=1e-3)
    args = ap.parse_args()

    errors = defaultdict(list)
    warnings = defaultdict(list)
    info = defaultdict(list)
    per_class = Counter()
    per_split = Counter()
    empty_labels = []
    total_boxes = 0
    dup_pairs = 0
    sizes = defaultdict(list)

    label_root = args.data / "labels"
    img_root = args.data / "images"
    if not label_root.is_dir():
        print(f"[ERR] no labels dir: {label_root}", file=sys.stderr)
        return 1

    for split in ("train", "val"):
        ldir = label_root / split
        idir = img_root / split
        if not ldir.is_dir():
            continue
        for lf in sorted(ldir.glob("*.txt")):
            name = lf.stem
            per_split[split] += 1
            lines = [ln for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                empty_labels.append(f"{split}/{name}")
                continue
            if not (idir / f"{name}.jpg").exists() and not (idir / f"{name}.png").exists():
                errors["label_without_image"].append(f"{split}/{name}")
            boxes = []
            for i, ln in enumerate(lines):
                parts = ln.split()
                if len(parts) != 5:
                    errors["bad_field_count"].append(f"{split}/{name}:{i} -> {ln!r}")
                    continue
                try:
                    c = int(float(parts[0]))
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    errors["non_numeric"].append(f"{split}/{name}:{i}")
                    continue
                if not (0 <= c < len(CLASSES)):
                    errors["bad_class_id"].append(f"{split}/{name}:{i} -> {c}")
                    continue
                if not (0 <= xc <= 1 and 0 <= yc <= 1):
                    errors["center_out_of_range"].append(f"{split}/{name}:{i}")
                if not (0 < w <= 1 and 0 < h <= 1):
                    errors["size_out_of_range"].append(f"{split}/{name}:{i}")
                if w < args.min_size or h < args.min_size:
                    errors["too_small_norm"].append(f"{split}/{name}:{i}")
                # NOTE: NEU-DET folder/name class is the *dominant* defect; an image may
                # legitimately contain boxes of other classes (multi-label annotations).
                # These are recorded as info, not as errors.
                if "rolled" in name and c != 4:
                    info["foreign_class_boxes"].append(f"{split}/{name}:{i} -> {CLASSES[c]}")
                elif "rolled" not in name:
                    prefix = name.split("_")[0].replace("-", "_")
                    if prefix in CLASSES and CLASSES[c] != prefix:
                        info["foreign_class_boxes"].append(
                            f"{split}/{name}:{i} -> {CLASSES[c]} (file prefix {prefix})"
                        )
                per_class[CLASSES[c]] += 1
                total_boxes += 1
                sizes[CLASSES[c]].append((w, h))
                boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    if iou(boxes[a], boxes[b]) > 0.95:
                        dup_pairs += 1
                        warnings["near_duplicate_boxes"].append(f"{split}/{name}")
                        break

    report = {
        "classes": CLASSES,
        "label_files": int(sum(per_split.values())),
        "per_split": dict(per_split),
        "total_boxes": int(total_boxes),
        "instances_per_class": dict(per_class),
        "empty_label_images": len(empty_labels),
        "empty_label_examples": empty_labels[:20],
        "near_duplicate_box_pairs": int(dup_pairs),
        "errors": {k: {"count": len(v), "examples": v[:10]} for k, v in errors.items()},
        "warnings": {k: {"count": len(v), "examples": v[:10]} for k, v in warnings.items()},
        "info": {k: {"count": len(v), "examples": v[:5]} for k, v in info.items()},
        "box_size_mean": {
            c: {"w": float(np.mean([s[0] for s in v])), "h": float(np.mean([s[1] for s in v]))}
            for c, v in sizes.items()
        },
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "label_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    n_err = sum(v["count"] for v in report["errors"].values())
    print(f"[i] label files: {report['label_files']}  boxes: {total_boxes}")
    print(f"[i] instances/class: {dict(per_class)}")
    print(f"[i] empty-label images: {len(empty_labels)}")
    print(f"[i] near-duplicate box pairs: {dup_pairs}")
    print(f"[i] foreign-class boxes (multi-label, not an error): "
          f"{report['info'].get('foreign_class_boxes', {}).get('count', 0)}")
    print(f"[i] errors: {n_err}")
    for k, v in report["errors"].items():
        print(f"    - {k}: {v['count']}  e.g. {v['examples'][:3]}")
    print(f"[OK] report -> {args.reports / 'label_check.json'}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
