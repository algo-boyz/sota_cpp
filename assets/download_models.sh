#!/usr/bin/env bash
# Download ONNX models from Hugging Face OpenCV Zoo (one-time).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODELS_DIR}"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    echo "Already present: ${out}"
    return
  fi
  echo "Downloading $(basename "${out}") …"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
  else
    echo "Need curl or wget to download models." >&2
    exit 1
  fi
}

# Hugging Face resolve URLs (public, no auth needed for these repos)
BASE_FER="https://huggingface.co/opencv/facial_expression_recognition/resolve/main"
BASE_YUNET="https://huggingface.co/opencv/face_detection_yunet/resolve/main"

download \
  "${BASE_FER}/facial_expression_recognition_mobilefacenet_2022july.onnx" \
  "${MODELS_DIR}/facial_expression_recognition_mobilefacenet_2022july.onnx"

download \
  "${BASE_YUNET}/face_detection_yunet_2023mar.onnx" \
  "${MODELS_DIR}/face_detection_yunet_2023mar.onnx"

echo "Done. Models are in ${MODELS_DIR}"
