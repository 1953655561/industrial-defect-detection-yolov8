#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect every measured number into resume_work/cv_metrics.json.

The resume builder reads this file, so every figure in the resume is traceable to
a report in reports/ or runs/. Re-run after any metric changes:

    python scripts/collect_metrics.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
REPORTS = HERE / "reports"
RUNS = HERE / "runs" / "detect"
OUT = Path(r"E:\xz\dsh work place\resume_work\cv_metrics.json")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def load(name: str) -> dict:
    p = REPORTS / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def epoch_metrics(run: str, epoch: int | None = None) -> dict:
    csv = RUNS / run / "results.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    if df.empty:
        return {}
    if epoch is not None:
        rows = df[df["epoch"] == epoch]
        row = rows.iloc[0] if len(rows) else df.iloc[-1]
    else:
        row = df.loc[df["metrics/mAP50(B)"].idxmax()]
    return {
        "epoch": int(row["epoch"]),
        "mAP50": float(row["metrics/mAP50(B)"]),
        "mAP50_95": float(row["metrics/mAP50-95(B)"]),
        "precision": float(row["metrics/precision(B)"]),
        "recall": float(row["metrics/recall(B)"]),
        "train_seconds": float(row["time"]) if "time" in df.columns else float("nan"),
        "epochs_logged": int(len(df)),
    }


def model_params_m(weights: Path) -> str:
    """Parameter count in millions from a checkpoint (needs torch, called after training)."""
    try:
        import torch

        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model = ckpt.get("model")
        if model is None:
            return ""
        n = sum(p.numel() for p in model.parameters())
        return f"{n / 1e6:.1f}"
    except Exception:
        return ""


def fmt3(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) and not math.isnan(v) else "—"


def fmt1(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) and not math.isnan(v) else "—"


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join([" --- "] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main() -> int:
    out: dict[str, str] = {}

    # ---- final metrics: prefer an explicit eval report, else best-epoch training log ----
    for run, key in (("y8n_aug", "y8n"), ("y8s_aug", "y8s"), ("y8n_noaug", "y8n_noaug")):
        ev = load(f"eval_{run}.json")
        if ev and ev.get("overall"):
            o = ev["overall"]
            vals = {"mAP50": o.get("mAP50"), "mAP50_95": o.get("mAP50_95"),
                    "precision": o.get("precision"), "recall": o.get("recall")}
        else:
            vals = epoch_metrics(run)
        if vals:
            out[f"{key}_map50"] = fmt3(vals.get("mAP50"))
            out[f"{key}_map50_95"] = fmt3(vals.get("mAP50_95"))
            out[f"{key}_p"] = fmt3(vals.get("precision"))
            out[f"{key}_r"] = fmt3(vals.get("recall"))
        ep = epoch_metrics(run)
        if ep:
            out[f"{key}_epochs"] = str(ep.get("epochs_logged", ""))
            out[f"{key}_best_epoch"] = str(ep.get("epoch", ""))
        # total wall time of the training run (train_*.json), not the time at best epoch
        tr = load(f"train_{run}.json")
        if tr.get("wall_seconds"):
            out[f"{key}_train_min"] = fmt1(float(tr["wall_seconds"]) / 60.0)
        w = RUNS / run / "weights" / "best.pt"
        if w.exists():
            pm = model_params_m(w)
            if pm:
                out[f"{key}_params_m"] = pm

    # ---- augmentation ablation (compare best checkpoints; also same-epoch for reference) ----
    aug_ev = load("eval_y8n_aug.json").get("overall", {})
    noaug_ev = load("eval_y8n_noaug.json").get("overall", {})
    e60n = epoch_metrics("y8n_aug", 60)
    e60no = epoch_metrics("y8n_noaug", 60)
    out["aug60_map50"] = fmt3(e60n.get("mAP50"))
    out["noaug60_map50"] = fmt3(e60no.get("mAP50"))
    aug_best = aug_ev.get("mAP50")
    noaug_best = noaug_ev.get("mAP50")
    if aug_best is not None and noaug_best is not None:
        out["aug_gain"] = f"{aug_best - noaug_best:+.3f}"
        out["noaug_best_epoch"] = str(e60no.get("epoch", "")) if e60no else ""
    # best epoch of the no-aug run (it peaks early then overfits)
    noaug_csv = RUNS / "y8n_noaug" / "results.csv"
    if noaug_csv.exists():
        d = pd.read_csv(noaug_csv)
        d.columns = [c.strip() for c in d.columns]
        if not d.empty:
            bi = d["metrics/mAP50(B)"].idxmax()
            out["noaug_best_epoch"] = str(int(d.loc[bi, "epoch"]))
            out["noaug_best_mAP50"] = fmt3(float(d.loc[bi, "metrics/mAP50(B)"]))
            out["noaug_last_mAP50"] = fmt3(float(d["metrics/mAP50(B)"].iloc[-1]))

    # YOLOv8s is trained shorter (30 epochs) because it is ~2x slower on the 8GB laptop GPU;
    # compare it with YOLOv8n at the *same* epoch (from the 100-epoch run's log).
    e30n = epoch_metrics("y8n_aug", 30)
    e30s = epoch_metrics("y8s_aug", 30)
    out["y8s30_map50"] = fmt3(e30s.get("mAP50"))
    out["y8n30_map50"] = fmt3(e30n.get("mAP50"))
    out["y8s_epochs"] = str(e30s.get("epochs_logged", ""))
    if e30s.get("mAP50") is not None and e30n.get("mAP50") is not None:
        out["y8s_vs_y8n_delta"] = f"{e30s['mAP50'] - e30n['mAP50']:+.3f}"
    out["ablation_epoch"] = "60"

    # ---- ONNX ----
    bench = load("onnx_bench.json")
    if bench:
        sr = bench.get("session_run", {})
        e2e = bench.get("end_to_end", {})
        out["onnx_ms"] = fmt1(sr.get("mean_ms"))
        out["onnx_p50"] = fmt1(sr.get("p50_ms"))
        out["onnx_p95"] = fmt1(sr.get("p95_ms"))
        out["onnx_fps"] = fmt1(sr.get("fps_from_mean"))
        out["e2e_ms"] = fmt1(e2e.get("mean_ms"))
        out["onnx_size_mb"] = fmt1(bench.get("model_size_mb"))
        cc = bench.get("cross_check", {})
        out["onnx_cross_check_agree"] = str(cc.get("all_agree", ""))
        out["onnx_cross_check_ratio"] = fmt3(cc.get("matched_ratio"))

    # ---- Linux (Docker / Debian) validation of the same ONNX model ----
    lin = load("onnx_bench_linux.json")
    if lin:
        lsr = lin.get("session_run", {})
        out["onnx_linux_ms"] = fmt1(lsr.get("mean_ms"))
        out["onnx_linux_p95"] = fmt1(lsr.get("p95_ms"))
        out["onnx_linux_e2e_ms"] = fmt1(lin.get("end_to_end", {}).get("mean_ms"))
        out["linux_validated"] = "Linux Debian 13 容器（WSL2 内核）"

    # ---- OpenCV baseline ----
    ocv = load("opencv_baseline.json")
    if ocv:
        out["ocv_p"] = fmt3(ocv.get("precision"))
        out["ocv_r"] = fmt3(ocv.get("recall"))
        out["ocv_f1"] = fmt3(ocv.get("f1"))
        out["ocv_fp"] = fmt3(ocv.get("fp_per_image")) if isinstance(ocv.get("fp_per_image"), float) else "—"
        out["ocv_ms"] = fmt1(ocv.get("mean_ms_per_image"))
        out["ocv_params"] = str(ocv.get("grid_search", {}).get("configs_tried", "")) + " 组"

    # ---- dataset ----
    ds = load("dataset_analysis.json")
    if ds:
        out["ds_images"] = str(ds.get("images_total"))
        out["ds_train"] = str(ds.get("split", {}).get("train"))
        out["ds_val"] = str(ds.get("split", {}).get("val"))
        out["ds_classes"] = str(len(ds.get("classes", [])))
        out["ds_boxes"] = str(ds.get("instances_total"))
        out["ds_multi_class_imgs"] = str(ds.get("images_with_multiple_classes"))
        out["ds_dup_groups"] = str(len(ds.get("duplicate_image_content", {})))

    # ---- markdown tables for README ----
    run_rows = []
    for run, label in (("y8n_aug", "YOLOv8n + 默认增强（主模型）"),
                       ("y8n_noaug", "YOLOv8n + 关闭增强（消融）"),
                       ("y8s_aug", "YOLOv8s + 默认增强（对比）")):
        ev = load(f"eval_{run}.json")
        ep = epoch_metrics(run)
        if not ep and not ev:
            continue
        o = (ev or {}).get("overall", {})
        m50 = o.get("mAP50", ep.get("mAP50"))
        m5095 = o.get("mAP50_95", ep.get("mAP50_95"))
        pp = o.get("precision", ep.get("precision"))
        rr = o.get("recall", ep.get("recall"))
        run_rows.append([label, ep.get("epochs_logged", "—"), ep.get("epoch", "—"),
                         fmt3(m50), fmt3(m5095), fmt3(pp), fmt3(rr),
                         (out.get(f"{'y8n' if run == 'y8n_aug' else run}_params_m", "—") or "—")])
    if run_rows:
        out["runs_table"] = md_table(
            ["模型 / 设置", "epoch", "最佳 epoch", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "参数量(M)"],
            run_rows,
        )

    ev = load("eval_y8n_aug.json")
    if ev and ev.get("per_class"):
        inst = ds.get("instances_per_class", {}) if ds else {}
        rows = []
        for cls, v in ev["per_class"].items():
            name = CLASSES[int(cls)] if str(cls).isdigit() and int(cls) < len(CLASSES) else cls
            rows.append([name, inst.get(name, "—"),
                         fmt3(v.get("precision")), fmt3(v.get("recall")),
                         fmt3(v.get("mAP50")), fmt3(v.get("mAP50_95"))])
        out["per_class_table"] = md_table(
            ["类别", "标注框数", "P", "R", "mAP@0.5", "mAP@0.5:0.95"], rows)

    ocv = load("opencv_baseline.json")
    if ocv and ocv.get("per_class_recall"):
        rows = []
        for cls, rec in ocv["per_class_recall"].items():
            rows.append([cls, ocv.get("per_class_gt", {}).get(cls, "—"), fmt3(rec)])
        out["ocv_table"] = md_table(["类别", "GT 框数", "传统视觉召回率"], rows)

    # ---- camera calibration (real photos only; synthetic is NOT used in the resume) ----
    calib = load("camera_calibration.json")
    if calib and calib.get("mode") == "real_photos":
        out["calib_rms"] = f"{calib.get('rms_reprojection_error_px', 0):.3f}"
        out["calib_views"] = str(calib.get("n_views"))
    else:
        out["calib_rms"] = ""
        out["calib_views"] = ""
    # real-photo validation on the public OpenCV sample set (not the applicant's own camera)
    sample = load("camera_calibration_sample.json")
    if sample:
        out["calib_sample_rms"] = f"{sample.get('rms_reprojection_error_px', 0):.3f}"
        out["calib_sample_views"] = str(sample.get("n_views", ""))
        out["calib_sample_mean_px"] = f"{sample.get('mean_per_view_error_px', 0):.3f}"
        out["calib_sample_max_px"] = f"{sample.get('max_per_view_error_px', 0):.3f}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
