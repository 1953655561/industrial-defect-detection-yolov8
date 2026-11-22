#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a printable A4 chessboard for camera calibration.

Default: 9x6 inner corners, 20 mm squares -> board is 200 x 140 mm, fits A4 with margins.
Saved as PNG (300 dpi) and PDF (A4).
Print at 100% / "actual size" (do NOT use "fit to page"), then measure one square with
a ruler; pass the measured size to camera_calibration.py --square.

    python scripts/make_chessboard.py
    python scripts/make_chessboard.py --square-mm 25 --cols 7 --rows 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
MM_PER_INCH = 25.4
A4_W_MM, A4_H_MM = 210.0, 297.0
PAGE_MARGIN_MM = 8.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="inner corners along width")
    ap.add_argument("--rows", type=int, default=6, help="inner corners along height")
    ap.add_argument("--square-mm", type=float, default=18.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=HERE / "assets")
    args = ap.parse_args()

    board_w_mm = (args.cols + 1) * args.square_mm
    board_h_mm = (args.rows + 1) * args.square_mm
    max_w = A4_W_MM - 2 * PAGE_MARGIN_MM
    max_h = A4_H_MM - 2 * PAGE_MARGIN_MM
    if board_w_mm > max_w or board_h_mm > max_h:
        max_sq = min(max_w / (args.cols + 1), max_h / (args.rows + 1))
        print(f"[ERR] board {board_w_mm:.1f}x{board_h_mm:.1f} mm does not fit A4 "
              f"(usable {max_w:.1f}x{max_h:.1f} mm)", file=sys.stderr)
        print(f"      use --square-mm <= {max_sq:.0f} or fewer squares", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sq_px = int(round(args.square_mm / MM_PER_INCH * args.dpi))
    w = (args.cols + 1) * sq_px
    h = (args.rows + 1) * sq_px
    arr = np.full((h, w), 255, np.uint8)
    for r in range(args.rows + 1):
        for c in range(args.cols + 1):
            if (r + c) % 2 == 0:
                arr[r * sq_px:(r + 1) * sq_px, c * sq_px:(c + 1) * sq_px] = 0
    img = Image.fromarray(arr, mode="L").convert("RGB")

    a4_w = int(round(A4_W_MM / MM_PER_INCH * args.dpi))
    a4_h = int(round(A4_H_MM / MM_PER_INCH * args.dpi))
    page = Image.new("RGB", (a4_w, a4_h), "white")
    ox, oy = (a4_w - w) // 2, (a4_h - h) // 2
    page.paste(img, (ox, oy))

    tag = f"chessboard_{args.cols}x{args.rows}_{int(args.square_mm)}mm_A4"
    png = args.out_dir / f"{tag}.png"
    pdf = args.out_dir / f"{tag}.pdf"
    page.save(png, dpi=(args.dpi, args.dpi))
    page.save(pdf, "PDF", resolution=args.dpi)

    print(f"[OK] {png}")
    print(f"[OK] {pdf}")
    print(f"[i] inner corners: {args.cols}x{args.rows}, square {args.square_mm} mm, "
          f"board {board_w_mm:.1f} x {board_h_mm:.1f} mm (A4 usable {max_w:.0f} x {max_h:.0f} mm)")
    print("[i] print at 100% on A4, glue to a flat rigid surface, measure one square with a ruler")
    print(f"[i] then: python scripts/camera_calibration.py --square {args.square_mm:g} "
          f"--pattern {args.cols}x{args.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
