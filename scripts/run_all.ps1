# One-shot reproduction script (Windows PowerShell).
# Usage:
#   .\scripts\run_all.ps1            # full pipeline
#   .\scripts\run_all.ps1 -SkipTrain # reuse existing runs/detect/*/weights/best.pt
param(
    [switch]$SkipTrain,
    [string]$Python = "E:\xz\cv_defect_det\.venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "== 1/8 prepare dataset =="
& $Python scripts/prepare_neu_det.py
& $Python scripts/verify_labels.py

if (-not $SkipTrain) {
    Write-Host "== 2/8 train yolov8n (augmented, 100 epochs) =="
    & $Python scripts/train.py --model yolov8n.pt --epochs 100 --name y8n_aug
    Write-Host "== 3/8 train yolov8s (augmented, 100 epochs) =="
    & $Python scripts/train.py --model yolov8s.pt --epochs 100 --name y8s_aug
    Write-Host "== 4/8 train yolov8n (no augmentation, ablation) =="
    & $Python scripts/train.py --model yolov8n.pt --epochs 100 --name y8n_noaug --aug none
}

Write-Host "== 5/8 evaluate + export ONNX =="
& $Python scripts/eval_yolo.py --weights runs/detect/y8n_aug/weights/best.pt --name y8n_aug
& $Python scripts/eval_yolo.py --weights runs/detect/y8s_aug/weights/best.pt --name y8s_aug
& $Python scripts/eval_yolo.py --weights runs/detect/y8n_noaug/weights/best.pt --name y8n_noaug
& $Python scripts/export_onnx.py --weights runs/detect/y8n_aug/weights/best.pt --name y8n_aug_onnx

Write-Host "== 6/8 ONNX latency benchmark =="
& $Python scripts/onnx_infer.py --model models/y8n_aug_onnx.onnx `
    --weights runs/detect/y8n_aug/weights/best.pt --bench --runs 100

Write-Host "== 7/8 OpenCV traditional baseline =="
& $Python scripts/opencv_baseline.py --yolo-eval reports/eval_y8n_aug.json

Write-Host "== 8/8 figures =="
& $Python scripts/make_figures.py --weights runs/detect/y8n_aug/weights/best.pt

Write-Host "done. metrics in reports/, figures in screenshots/"
