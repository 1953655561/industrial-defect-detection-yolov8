#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chessboard camera calibration (Zhang's method) with OpenCV.

Real-photo mode (default):
    put 10-15 chessboard photos in assets/calib_photos/ (any resolution, different angles,
    the whole board visible), then:
        python scripts/camera_calibration.py

Self-test mode (no photos needed):
        python scripts/camera_calibration.py --synthetic
    renders synthetic chessboard views with a known focal length and runs the same
    detection + calibration code path, so the pipeline itself is verified. This is a
    code self-test only: it must NOT be reported as a real camera calibration.

Outputs reports/camera_calibration.json plus an undistortion before/after figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent


def make_board(inner=(9, 6), square_px=100) -> np.ndarray:
    cols, rows = inner
    board = np.zeros(((rows + 1) * square_px, (cols + 1) * square_px), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * square_px:(r + 1) * square_px,
                      c * square_px:(c + 1) * square_px] = 255
    return board


def find_corners(gray, inner):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if hasattr(cv2, "findChessboardCornersSB"):
        # SB only accepts its own flags (EXHAUSTIVE / ACCURACY / LARGER / MARKER)
        ok, corners = cv2.findChessboardCornersSB(
            gray, inner, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        )
        if ok:
            return True, corners.reshape(-1, 1, 2).astype(np.float32)
    ok, corners = cv2.findChessboardCorners(gray, inner, flags=flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def build_object_points(inner, square):
    cols, rows = inner
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts * float(square)


def calibrate(views, inner, square):
    """views: list of (gray_like, corners). Returns calibration dict."""
    objp = build_object_points(inner, square)
    objpoints, imgpoints, used = [], [], []
    for gray, corners in views:
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used.append(gray.shape[::-1])
    h, w = views[0][0].shape[:2]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None, flags=0
    )
    per_view = []
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        proj, _ = cv2.projectPoints(objpoints[i], rv, tv, K, dist)
        err = np.linalg.norm(imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        per_view.append({"view": i, "mean_err_px": float(err.mean()), "max_err_px": float(err.max())})
    return {
        "rms_reprojection_error_px": float(rms),
        "mean_per_view_error_px": float(np.mean([v["mean_err_px"] for v in per_view])),
        "max_per_view_error_px": float(np.max([v["max_err_px"] for v in per_view])),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "image_size_wh": [int(w), int(h)],
        "n_views": len(views),
        "per_view": per_view,
    }


def synthetic_views(inner=(9, 6), square=25.0, n=14, w=1280, h=960, seed=0):
    """Render n poses of a planar board under a known pinhole camera (no distortion)."""
    true_f = 900.0
    K_true = np.array([[true_f, 0, w / 2], [0, true_f, h / 2], [0, 0, 1]], np.float64)
    dist_true = np.zeros(5)
    cols, rows = inner
    board_pts = build_object_points(inner, square)
    board_img = make_board(inner, 100)
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n):
        # random moderate pose
        rvec = np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.3, 0.3),
        ])
        tvec = np.array([rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(380, 520)])
        proj, _ = cv2.projectPoints(board_pts, rvec, tvec, K_true, dist_true)
        proj = proj.reshape(-1, 2)
        # homography from board pixel coords (100px/square) to image
        src = board_pts[:, :2] / square * 100.0
        src = src.astype(np.float32)
        H, _ = cv2.findHomography(src, proj.astype(np.float32))
        warped = cv2.warpPerspective(board_img, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderValue=127)
        views.append(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))
    return views, K_true, dist_true


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=HERE / "assets" / "calib_photos")
    ap.add_argument("--pattern", default="9x6", help="inner corners cols x rows")
    ap.add_argument("--square", type=float, default=18.0,
                    help="square size in mm (must match the printed board)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "reports" / "camera_calibration.json")
    ap.add_argument("--screens", type=Path, default=HERE / "screenshots")
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    inner = (cols, rows)
    args.screens.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        imgs, K_true, dist_true = synthetic_views(inner, args.square)
        views = []
        for im in imgs:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ok, corners = find_corners(gray, inner)
            if ok:
                views.append((gray, corners))
        if len(views) < 3:
            print(f"[ERR] only {len(views)} synthetic views detected", file=sys.stderr)
            return 1
        calib = calibrate(views, inner, args.square)
        calib["mode"] = "synthetic_self_test"
        calib["note"] = ("Code self-test on rendered images with known pinhole camera and zero "
                         "distortion. NOT a real camera calibration and not resume evidence.")
        calib["ground_truth"] = {"fx": float(K_true[0, 0]), "fy": float(K_true[1, 1]),
                                 "cx": float(K_true[0, 2]), "cy": float(K_true[1, 2])}
        calib["recovered_vs_truth"] = {
            "fx_rel_error": abs(calib["camera_matrix"][0][0] - K_true[0, 0]) / K_true[0, 0],
            "fy_rel_error": abs(calib["camera_matrix"][1][1] - K_true[1, 1]) / K_true[1, 1],
            "cx_err_px": abs(calib["camera_matrix"][0][2] - K_true[0, 2]),
            "cy_err_px": abs(calib["camera_matrix"][1][2] - K_true[1, 2]),
        }
        print(json.dumps({k: calib[k] for k in ("n_views", "rms_reprojection_error_px",
                                                "mean_per_view_error_px", "recovered_vs_truth")}, indent=2))
        args.out.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] report -> {args.out}")
        return 0

    # ---------- real photos ----------
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(args.images.glob("*")) if p.suffix.lower() in exts] \
        if args.images.is_dir() else []
    if not files:
        print(f"[ERR] no calibration photos in {args.images}", file=sys.stderr)
        print("      1) run: python scripts/make_chessboard.py")
        print("      2) print the PDF on A4 at 100% scale, glue it to a rigid flat surface")
        print("      3) take 12-15 photos from different angles/distances with your phone/camera")
        print(f"      4) copy them to {args.images}")
        print("      5) rerun this script")
        return 2

    views = []
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray, inner)
        if ok:
            views.append((gray, corners))
        print(f"  {p.name}: {'OK' if ok else 'board not found'}")
    if len(views) < 3:
        print(f"[ERR] usable views: {len(views)} (<3)", file=sys.stderr)
        return 1

    calib = calibrate(views, inner, args.square)
    calib["mode"] = "real_photos"
    calib["square_size_mm"] = args.square
    calib["pattern_inner"] = list(inner)
    calib["source_images"] = [p.name for p in files]
    args.out.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")

    # undistort before/after on the first usable photo
    gray0 = views[0][0]
    h0, w0 = gray0.shape
    K = np.array(calib["camera_matrix"])
    D = np.array(calib["dist_coeffs"])
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w0, h0), 1, (w0, h0))
    undist = cv2.undistort(gray0, K, D, None, new_K)
    x, y, rw, rh = roi
    undist_c = undist[y:y + rh, x:x + rw] if rh > 0 and rw > 0 else undist
    undist_c = cv2.resize(undist_c, (w0, h0))
    fig = np.hstack([cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(undist_c, cv2.COLOR_GRAY2BGR)])
    cv2.putText(fig, "original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(fig, "undistorted", (w0 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(args.screens / "calibration_undistort.jpg"), fig)

    print(json.dumps({k: calib[k] for k in ("n_views", "rms_reprojection_error_px",
                                            "mean_per_view_error_px", "max_per_view_error_px")}, indent=2))
    print(f"[OK] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
