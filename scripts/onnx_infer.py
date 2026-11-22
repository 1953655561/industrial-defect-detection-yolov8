#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""onnxruntime inference for the exported YOLOv8 detector: decode + NMS + latency benchmark.

The ONNX graph exported by ultralytics is NMS-free: output0 has shape (1, 4+nc, 8400)
with xywh boxes in *input-image* pixels and sigmoided class scores. Post-processing
(decode, thresholding, NMS, rescale to the original image) is implemented here, so the
reported end-to-end latency includes the same work a deployment would do.

Also cross-checks ONNX detections against the PyTorch model on a few val images to prove
the exported graph is numerically equivalent (same detections, IoU >= 0.95).

Examples:
    python scripts/onnx_infer.py --model models/y8n_aug_onnx.onnx --weights runs/detect/y8n_aug/weights/best.pt --bench
    python scripts/onnx_infer.py --model models/y8n_aug_onnx.onnx --image data/neu_det/images/val/crazing_1.jpg
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent.parent
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


# --------------------------------------------------------------------------- pre-process
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    """Resize + pad keeping aspect ratio, ultralytics-compatible for inference."""
    h0, w0 = im.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (left, top)


def preprocess(path: Path, imgsz: int):
    im0 = cv2.imread(str(path))  # BGR
    if im0 is None:
        raise FileNotFoundError(path)
    im, ratio, pad = letterbox(im0, (imgsz, imgsz))
    x = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], im0, ratio, pad


# --------------------------------------------------------------------------- post-process
def decode(output: np.ndarray, ratio, pad, imgsz: int, conf_thres: float, iou_thres: float):
    """output: (1, 4+nc, N) -> boxes xyxy in original image coords, scores, class ids."""
    pred = output[0].T  # (N, 4+nc)
    xywh = pred[:, :4]
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf > conf_thres
    if not keep.any():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    xywh, conf, cls = xywh[keep], conf[keep], cls[keep]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    # undo letterbox: subtract padding, divide by ratio
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio[0]
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio[1]
    # class-aware NMS: shift boxes per class so NMS does not merge different classes
    off = cls.astype(np.float32) * (imgsz * 10.0)
    shifted = xyxy + off[:, None]
    boxes_list = shifted.tolist()
    idx = cv2.dnn.NMSBoxes(boxes_list, conf.tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    idx = np.asarray(idx).reshape(-1)
    boxes = xyxy[idx]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, None)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, None)
    return boxes.astype(np.float32), conf[idx].astype(np.float32), cls[idx].astype(np.int64)


# --------------------------------------------------------------------------- plotting
def draw(img, boxes, scores, cls, color=(0, 200, 0)):
    out = img.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        label = f"{CLASSES[c]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_session(path: str, provider: str):
    if provider == "auto":
        providers = ort.get_available_providers()
        providers = ["CUDAExecutionProvider"] + providers if "CUDAExecutionProvider" in providers else providers
    else:
        providers = [provider]
    return ort.InferenceSession(path, providers=providers)


def bench(sess, x, n_warmup: int, n_runs: int) -> dict:
    inp_name = sess.get_inputs()[0].name
    for _ in range(n_warmup):
        sess.run(None, {inp_name: x})
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        sess.run(None, {inp_name: x})
        times.append((time.perf_counter() - t) * 1000.0)
    times = np.array(times)
    return {
        "n_runs": int(n_runs),
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "fps_from_mean": float(1000.0 / times.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=HERE / "models" / "y8n_aug_onnx.onnx")
    ap.add_argument("--weights", type=Path, default=None,
                    help="optional .pt for numerical cross-check against ONNX")
    ap.add_argument("--images", type=Path, default=HERE / "data" / "neu_det" / "images" / "val")
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--check-n", type=int, default=8)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    sess = make_session(str(args.model), args.provider)
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name

    # --- single image ---
    if args.image is not None:
        x, im0, ratio, pad = preprocess(args.image, args.imgsz)
        out = sess.run(None, {inp.name: x})[0]
        boxes, scores, cls = decode(out, ratio, pad, args.imgsz, args.conf, args.iou)
        vis = draw(im0, boxes, scores, cls)
        dest = HERE / "screenshots"
        dest.mkdir(exist_ok=True)
        p = dest / f"onnx_infer_{args.image.stem}.jpg"
        cv2.imwrite(str(p), vis)
        print(f"[OK] {args.image} -> {len(boxes)} detections -> {p}")
        print(json.dumps([{"class": CLASSES[c], "score": round(float(s), 3),
                           "box": [round(float(v), 1) for v in b]}
                          for b, s, c in zip(boxes, scores, cls)], ensure_ascii=False))
        return 0

    # --- benchmark ---
    imgs = sorted(args.images.glob("*.jpg"))[: max(1, args.check_n)]
    x, im0, ratio, pad = preprocess(imgs[0], args.imgsz)
    report = {
        "model": str(args.model),
        "model_size_mb": round(args.model.stat().st_size / 1e6, 2),
        "input_shape": list(inp.shape),
        "output_shape": list(sess.get_outputs()[0].shape),
        "providers_used": sess.get_providers(),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
    }

    if args.bench:
        report["session_run"] = bench(sess, x, args.warmup, args.runs)
        # end-to-end: preprocess + infer + decode/NMS
        e2e = []
        for _ in range(args.runs):
            t = time.perf_counter()
            xx, _, rr, pp = preprocess(imgs[0], args.imgsz)
            out = sess.run(None, {inp.name: xx})[0]
            decode(out, rr, pp, args.imgsz, args.conf, args.iou)
            e2e.append((time.perf_counter() - t) * 1000.0)
        e2e = np.array(e2e)
        report["end_to_end"] = {
            "mean_ms": float(e2e.mean()),
            "p50_ms": float(np.percentile(e2e, 50)),
            "p95_ms": float(np.percentile(e2e, 95)),
            "fps_from_mean": float(1000.0 / e2e.mean()),
        }

    # --- numerical cross-check against PyTorch ---
    if args.weights is not None:
        from ultralytics import YOLO

        pt = YOLO(str(args.weights))
        rows = []
        for p in imgs:
            xx, im0, rr, pp = preprocess(p, args.imgsz)
            out = sess.run(None, {inp.name: xx})[0]
            b_onnx, s_onnx, c_onnx = decode(out, rr, pp, args.imgsz, 0.25, 0.45)
            r = pt.predict(source=str(p), imgsz=args.imgsz, conf=0.25, iou=0.45,
                           device="cpu", verbose=False)[0]
            b_pt = r.boxes.xyxy.cpu().numpy()
            c_pt = r.boxes.cls.cpu().numpy().astype(int)
            # greedy match by class + IoU
            matched = 0
            used = set()
            for i in range(len(b_onnx)):
                for j in range(len(b_pt)):
                    if j in used or c_pt[j] != c_onnx[i]:
                        continue
                    ax1, ay1, ax2, ay2 = b_onnx[i]
                    bx1, by1, bx2, by2 = b_pt[j]
                    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
                    ih = max(0, min(ay2, by2) - max(ay1, by1))
                    inter = iw * ih
                    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                    if ua > 0 and inter / ua >= 0.95:
                        matched += 1
                        used.add(j)
                        break
            rows.append({
                "image": p.name,
                "onnx_dets": int(len(b_onnx)),
                "pt_dets": int(len(b_pt)),
                "matched_iou95": matched,
                "agree": bool(len(b_onnx) == len(b_pt) and matched == len(b_onnx)),
            })
        report["cross_check"] = {
            "images": len(rows),
            "all_agree": all(r["agree"] for r in rows),
            "matched_ratio": (
                float(sum(r["matched_iou95"] for r in rows) / max(1, sum(r["onnx_dets"] for r in rows)))
            ),
            "detail": rows,
        }

    reports = HERE / "reports"
    reports.mkdir(exist_ok=True)
    out_json = args.out_json or (reports / "onnx_bench.json")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[OK] report -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
