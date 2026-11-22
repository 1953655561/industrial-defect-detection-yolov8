#!/usr/bin/env bash
# Linux-side validation of the exported ONNX detector.
#
# Runs the same inference/benchmark code as the Windows host, but inside a Linux
# container or a GitHub Actions ubuntu runner, proving the model is not tied to
# the training machine:
#
#   docker run --rm -v "$PWD:/work" -w /work python:3.12-slim bash scripts/validate_linux.sh
#
# The dataset is not redistributed, so if data/neu_det is missing the single-image
# step uses a tiny synthetically generated image (validation of the pipeline, not mAP).
# Outputs reports/onnx_bench_linux.json.
set -euo pipefail

echo "== linux environment =="
uname -a
python -V
head -n 3 /etc/os-release 2>/dev/null || true

echo "== install inference deps =="
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
pip install --quiet --disable-pip-version-check \
  onnxruntime numpy opencv-python-headless \
  -i "$PIP_INDEX"

python - <<'PY'
import sys
import cv2
import numpy
import onnxruntime
print("python      :", sys.version.split()[0])
print("opencv      :", cv2.__version__)
print("numpy       :", numpy.__version__)
print("onnxruntime :", onnxruntime.__version__)
print("providers   :", onnxruntime.get_available_providers())
PY

echo "== single image inference =="
IMG="data/neu_det/images/val/crazing_1.jpg"
if [ ! -f "$IMG" ]; then
  echo "(dataset not found, generating a synthetic 200x200 image)"
  python - <<'PY'
import cv2
import numpy as np
img = np.full((200, 200), 128, np.uint8)
cv2.line(img, (20, 180), (180, 20), 40, 3)
img = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imwrite("/tmp/synth_defect.jpg", img)
PY
  IMG=/tmp/synth_defect.jpg
fi
python scripts/onnx_infer.py --model models/y8n_aug_onnx.onnx --image "$IMG"

echo "== latency benchmark (CPU, 50 runs) =="
python scripts/onnx_infer.py \
  --model models/y8n_aug_onnx.onnx \
  --bench --runs 50 \
  --out-json reports/onnx_bench_linux.json

echo "LINUX VALIDATION OK"
