#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export a YOLOv8 checkpoint to ONNX and verify the exported graph runs.

Example:
    python scripts/export_onnx.py --weights runs/detect/y8n_aug/weights/best.pt --imgsz 640 --opset 12
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--simplify", action="store_true", default=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"{args.weights.parent.parent.name}_onnx"
    model = YOLO(str(args.weights))
    t0 = time.time()
    onnx_path = Path(
        model.export(format="onnx", opset=args.opset, simplify=args.simplify,
                     imgsz=args.imgsz, dynamic=False, batch=1)
    )
    export_s = time.time() - t0
    models_dir = HERE / "models"
    models_dir.mkdir(exist_ok=True)
    final = models_dir / f"{name}.onnx"
    shutil.copy2(onnx_path, final)

    sess = ort.InferenceSession(str(final), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    outs = sess.get_outputs()
    dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    y = sess.run(None, {inp.name: dummy})[0]

    info = {
        "onnx": str(final),
        "size_mb": round(final.stat().st_size / 1e6, 2),
        "export_seconds": round(export_s, 1),
        "opset": args.opset,
        "input": {"name": inp.name, "shape": inp.shape, "type": inp.type},
        "output": {"name": outs[0].name, "shape": list(y.shape), "type": outs[0].type},
        "providers": ort.get_available_providers(),
        "ort_version": ort.__version__,
    }
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "reports" / f"onnx_export_{name}.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
