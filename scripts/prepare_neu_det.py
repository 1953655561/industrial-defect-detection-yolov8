#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NEU-DET: VOC XML -> YOLO txt, stratified train/val split, dataset analysis.

Usage:
    python scripts/prepare_neu_det.py --raw data/raw/NEU-DET --out data/neu_det --seed 42 --val-ratio 0.2

Input layout (Kaggle mirror of the public NEU-DET database):
    raw/NEU-DET/train/images/<class>/<name>.jpg
    raw/NEU-DET/train/annotations/<name>.xml
    raw/NEU-DET/validation/...   (same shape)

The original Kaggle mirror ships its own train/validation folders. We do NOT trust
that split: all images are pooled and re-split per class with a fixed seed so the
split is reproducible and stratified. This is reported in README.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def parse_voc(xml_path: Path):
    """Return (width, height, [ {cls,xmin,ymin,xmax,ymax}, ... ])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    objs = []
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip()
        bb = obj.find("bndbox")
        objs.append(
            dict(
                cls=name,
                xmin=float(bb.find("xmin").text),
                ymin=float(bb.find("ymin").text),
                xmax=float(bb.find("xmax").text),
                ymax=float(bb.find("ymax").text),
            )
        )
    return w, h, objs


def voc_to_yolo(objs, w, h):
    """Convert to normalized YOLO lines. Returns (lines, n_kept, n_dropped)."""
    lines, kept, dropped = [], 0, 0
    for o in objs:
        if o["cls"] not in CLASS_TO_ID:
            dropped += 1
            continue
        xmin = min(max(o["xmin"], 0.0), w)
        xmax = min(max(o["xmax"], 0.0), w)
        ymin = min(max(o["ymin"], 0.0), h)
        ymax = min(max(o["ymax"], 0.0), h)
        bw, bh = xmax - xmin, ymax - ymin
        if bw < 1.0 or bh < 1.0:  # degenerate annotation
            dropped += 1
            continue
        xc, yc = (xmin + xmax) / 2.0 / w, (ymin + ymax) / 2.0 / h
        lines.append(
            f"{CLASS_TO_ID[o['cls']]} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}"
        )
        kept += 1
    return lines, kept, dropped


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect(raw: Path):
    images = []  # (path, kaggle_split, class_dir)
    for split in ("train", "validation"):
        img_root = raw / split / "images"
        if not img_root.is_dir():
            continue
        for p in sorted(img_root.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                images.append((p, split, p.parent.name))
    xmls = {}
    for split in ("train", "validation"):
        ann_root = raw / split / "annotations"
        if ann_root.is_dir():
            for p in sorted(ann_root.glob("*.xml")):
                xmls.setdefault(p.stem, []).append(p)
    return images, xmls


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--raw", type=Path, default=here / "data" / "raw" / "NEU-DET")
    ap.add_argument("--out", type=Path, default=here / "data" / "neu_det")
    ap.add_argument("--reports", type=Path, default=here / "reports")
    ap.add_argument("--screens", type=Path, default=here / "screenshots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    raw, out = args.raw.resolve(), args.out.resolve()
    if not raw.is_dir():
        print(f"[ERR] raw dataset not found: {raw}", file=sys.stderr)
        return 1
    args.reports.mkdir(parents=True, exist_ok=True)
    args.screens.mkdir(parents=True, exist_ok=True)

    images, xmls = collect(raw)
    print(f"[i] images found: {len(images)}  xml files: {sum(len(v) for v in xmls.values())}")

    # ---- anomalies: image w/o xml, xml w/o image, duplicate xml ----
    no_xml = [p for p, _, _ in images if p.stem not in xmls]
    img_stems = {p.stem for p, _, _ in images}
    no_img = sorted({s for s in xmls if s not in img_stems})
    dup_xml = {s: [str(p) for p in v] for s, v in xmls.items() if len(v) > 1}

    # ---- duplicate image content (md5) ----
    hash_groups = defaultdict(list)
    for p, _, _ in images:
        hash_groups[md5_of(p)].append(p.name)
    duplicates = {h: names for h, names in hash_groups.items() if len(names) > 1}
    name_to_md5 = {name: h for h, names in hash_groups.items() for name in names}

    # ---- parse annotations / build records ----
    records = []
    inst_counter = Counter()
    area_ratios = defaultdict(list)
    aspect = defaultdict(list)
    boxes_per_image = Counter()
    multi_class_imgs = []
    foreign_class_imgs = []
    dropped_boxes = 0
    for p, ksplit, cls_dir in images:
        stem = p.stem
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] unreadable image: {p}")
            continue
        h, w = img.shape[:2]
        objs = []
        if stem in xmls:
            dw, dh, objs = parse_voc(xmls[stem][0])
            if (dw, dh) != (w, h):
                print(f"[WARN] size mismatch {stem}: xml={dw}x{dh} img={w}x{h}")
                objs = parse_voc(xmls[stem][0])[2]
        yolo_lines, kept, drop = voc_to_yolo(objs, w, h)
        dropped_boxes += drop
        classes_in_img = {o["cls"] for o in objs if o["cls"] in CLASS_TO_ID}
        boxes_per_image[kept] += 1
        if len(classes_in_img) > 1:
            multi_class_imgs.append(stem)
        if classes_in_img and classes_in_img != {cls_dir}:
            foreign_class_imgs.append(stem)
        for o in objs:
            if o["cls"] in CLASS_TO_ID:
                inst_counter[o["cls"]] += 1
                bw = max(o["xmax"] - o["xmin"], 1.0)
                bh = max(o["ymax"] - o["ymin"], 1.0)
                area_ratios[o["cls"]].append(bw * bh / (w * h))
                aspect[o["cls"]].append(bw / bh)
        records.append(
            dict(
                stem=stem,
                img=str(p),
                w=w,
                h=h,
                cls_dir=cls_dir,
                n_boxes=kept,
                yolo_lines=yolo_lines,
                kaggle_split=ksplit,
                md5=name_to_md5.get(p.name, ""),
            )
        )

    # ---- stratified split (per class, deterministic) ----
    rng = random.Random(args.seed)
    by_class = defaultdict(list)
    for r in records:
        key = r["cls_dir"] if r["cls_dir"] in CLASS_TO_ID else "unknown"
        by_class[key].append(r)
    split_out = {}
    for cls in sorted(by_class):
        group = sorted(by_class[cls], key=lambda r: r["stem"])
        rng.shuffle(group)
        n_val = int(round(len(group) * args.val_ratio))
        # identical-content images (same md5) must stay in the same split
        by_hash = defaultdict(list)
        for r in group:
            by_hash[r["md5"] or r["stem"]].append(r)
        buckets = list(by_hash.values())
        rng.shuffle(buckets)
        val, train = [], []
        for b in buckets:
            if len(val) + len(b) <= n_val:
                val.extend(b)
            else:
                train.extend(b)
        for r in train:
            split_out[r["stem"]] = "train"
        for r in val:
            split_out[r["stem"]] = "val"

    # ---- write YOLO tree ----
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for r in records:
        s = split_out[r["stem"]]
        shutil.copy2(r["img"], out / "images" / s / f"{r['stem']}.jpg")
        (out / "labels" / s / f"{r['stem']}.txt").write_text(
            "\n".join(r["yolo_lines"]) + ("\n" if r["yolo_lines"] else ""), encoding="utf-8"
        )

    yaml_path = here / "data" / "neu_det.yaml"
    yaml_path.write_text(
        "# Auto-generated by scripts/prepare_neu_det.py\n"
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
        encoding="utf-8",
    )

    # ---- analysis json ----
    n_train = sum(1 for s in split_out.values() if s == "train")
    n_val = sum(1 for s in split_out.values() if s == "val")
    analysis = {
        "dataset": "NEU-DET (public, Northeastern University; Kaggle mirror kaustubhdikshit/neu-surface-defect-database)",
        "classes": CLASSES,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "images_total": len(records),
        "split": {"train": n_train, "val": n_val},
        "split_per_class": {
            c: {
                "train": sum(1 for r in records if r["cls_dir"] == c and split_out[r["stem"]] == "train"),
                "val": sum(1 for r in records if r["cls_dir"] == c and split_out[r["stem"]] == "val"),
            }
            for c in CLASSES
        },
        "instances_total": int(sum(inst_counter.values())),
        "instances_per_class": {c: int(inst_counter[c]) for c in CLASSES},
        "boxes_dropped_as_degenerate_or_unknown": dropped_boxes,
        "boxes_per_image": {str(k): int(v) for k, v in sorted(boxes_per_image.items())},
        "images_with_multiple_classes": len(multi_class_imgs),
        "images_with_foreign_class_boxes": len(foreign_class_imgs),
        "multi_class_note": ("NEU-DET images are grouped into 6 folders by dominant defect, but some "
                             "images contain additional defect types in the annotations; these are "
                             "genuine multi-label annotations, not label errors."),
        "images_without_xml": [p.name for p in no_xml],
        "xml_without_image": no_img,
        "duplicate_annotation_files": dup_xml,
        "duplicate_image_content": duplicates,
        "duplicate_groups_split_leak": sum(
            1 for names in duplicates.values()
            if len({split_out.get(Path(n).stem) for n in names}) > 1
        ),
        "box_area_ratio": {
            c: {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
            }
            for c, v in area_ratios.items()
        },
        "box_aspect_wh": {
            c: {"mean": float(np.mean(v)), "median": float(np.median(v))}
            for c, v in aspect.items()
        },
    }
    (args.reports / "dataset_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- markdown summary ----
    lines = [
        "# NEU-DET 数据集分析",
        "",
        f"- 图像总数：**{len(records)}**（train {n_train} / val {n_val}，按类别分层，seed={args.seed}）",
        f"- 标注框总数：**{int(sum(inst_counter.values()))}**",
        f"- 每类图像数：300（6 类，公开数据集原始分布）",
        "",
        "| 类别 | train 图 | val 图 | 标注框数 | 框面积占比中位数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in CLASSES:
        lines.append(
            f"| {c} | {analysis['split_per_class'][c]['train']} | "
            f"{analysis['split_per_class'][c]['val']} | {inst_counter[c]} | "
            f"{np.median(area_ratios[c]):.3f} |"
        )
    lines += [
        "",
        "## 数据质量检查",
        f"- 含多类缺陷的图：{len(multi_class_imgs)} 张（标注里出现非主类别框，属 NEU-DET 原始多标签标注，非错标）",
        f"- 无 XML 的图像：{len(no_xml)}（{', '.join(p.name for p in no_xml) or '无'}）-> 按背景图处理，标签为空文件",
        f"- 无图像的 XML：{len(no_img)}（{', '.join(no_img) or '无'}）",
        f"- 重复标注文件：{len(dup_xml)}",
        f"- 内容重复（MD5）图像组：{len(duplicates)}（同组图像强制划分到同一 split，避免泄漏）",
        f"- 丢弃的退化/未知类别框：{dropped_boxes}",
        "",
        "> 划分说明：本划分是对全部 1800 张图像重新做的分层随机划分，未沿用镜像自带的 split；"
        "未按钢板/批次分组，因此同一母卷图像可能同时出现在 train 与 val，存在同源泄漏风险（见 README 局限）。",
    ]
    (args.reports / "dataset_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- GT visualization ----
    vis_rng = random.Random(args.seed + 1)
    picks = [r for r in records]
    vis_rng.shuffle(picks)
    tiles = []
    for r in picks[:12]:
        img = cv2.imread(r["img"], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in r["yolo_lines"]:
            c, xc, yc, bw, bh = line.split()
            c = int(c)
            x1 = int((float(xc) - float(bw) / 2) * r["w"])
            y1 = int((float(yc) - float(bh) / 2) * r["h"])
            x2 = int((float(xc) + float(bw) / 2) * r["w"])
            y2 = int((float(yc) + float(bh) / 2) * r["h"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, CLASSES[c], (x1 + 1, max(y1 + 8, 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 0), 1, cv2.LINE_AA)
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
    cv2.imwrite(str(args.screens / "dataset_gt_samples.jpg"), np.vstack(rows))

    print(json.dumps(analysis["split"], ensure_ascii=False))
    print(f"[OK] yolo dataset -> {out}")
    print(f"[OK] yaml -> {yaml_path}")
    print(f"[OK] reports -> {args.reports / 'dataset_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
