#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

# Store huge archives on Linux filesystem by default (more reliable than /mnt/d).
HF_RAW_DIR="${HF_RAW_DIR:-${HOME}/didemo_cache/hf_raw}"
HF_BASE_URL="https://huggingface.co/datasets/YimuWang/didemon_retrieval/resolve/main"

mkdir -p "${ROOT_DIR}/snapshots" "${ROOT_DIR}/prototxts" "${DATA_DIR}" "${HF_RAW_DIR}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: '${cmd}' is required but not found."
    exit 1
  fi
}

download_glove() {
  pushd "${DATA_DIR}" >/dev/null
  if [ ! -f "glove.6B.zip" ]; then
    wget -c "http://nlp.stanford.edu/data/glove.6B.zip"
  fi
  unzip -o "glove.6B.zip"
  rm -f "glove.6B.50d.txt" "glove.6B.100d.txt" "glove.6B.200d.txt"
  popd >/dev/null
}

try_primary_average_download() {
  local ok=1

  if [ ! -f "${DATA_DIR}/average_fc7.h5" ]; then
    if ! wget -c -O "${DATA_DIR}/average_fc7.h5" \
      "https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5"; then
      rm -f "${DATA_DIR}/average_fc7.h5"
      ok=0
    fi
  fi

  if [ ! -f "${DATA_DIR}/average_global_flow.h5" ]; then
    if ! wget -c -O "${DATA_DIR}/average_global_flow.h5" \
      "https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5"; then
      rm -f "${DATA_DIR}/average_global_flow.h5"
      ok=0
    fi
  fi

  if [ "${ok}" -eq 1 ] && \
    [ -f "${DATA_DIR}/average_fc7.h5" ] && \
    [ -f "${DATA_DIR}/average_global_flow.h5" ]; then
    return 0
  fi

  return 1
}

download_hf_parts() {
  pushd "${HF_RAW_DIR}" >/dev/null

  for i in 0 1 2 3; do
    wget -c "${HF_BASE_URL}/flow_features.tar.${i}"
  done

  for i in $(seq -w 0 12); do
    wget -c "${HF_BASE_URL}/rgb_vgg_fc7_features.tar.${i}"
  done

  popd >/dev/null
}

assemble_hf_archives() {
  pushd "${HF_RAW_DIR}" >/dev/null

  cat flow_features.tar.0 flow_features.tar.1 flow_features.tar.2 flow_features.tar.3 > flow_features.tar
  tar -tf flow_features.tar >/dev/null

  cat \
    rgb_vgg_fc7_features.tar.00 \
    rgb_vgg_fc7_features.tar.01 \
    rgb_vgg_fc7_features.tar.02 \
    rgb_vgg_fc7_features.tar.03 \
    rgb_vgg_fc7_features.tar.04 \
    rgb_vgg_fc7_features.tar.05 \
    rgb_vgg_fc7_features.tar.06 \
    rgb_vgg_fc7_features.tar.07 \
    rgb_vgg_fc7_features.tar.08 \
    rgb_vgg_fc7_features.tar.09 \
    rgb_vgg_fc7_features.tar.10 \
    rgb_vgg_fc7_features.tar.11 \
    rgb_vgg_fc7_features.tar.12 > rgb_vgg_fc7_features.tar
  tar -tf rgb_vgg_fc7_features.tar >/dev/null

  if [ ! -d flow_features ]; then
    tar -xf flow_features.tar
  fi
  if [ ! -d rgb_vgg_fc7_features ]; then
    tar -xf rgb_vgg_fc7_features.tar
  fi

  popd >/dev/null
}

build_average_features_from_hf() {
  pushd "${ROOT_DIR}" >/dev/null
  ln -sfn "${HF_RAW_DIR}/flow_features" flow_features
  ln -sfn "${HF_RAW_DIR}/rgb_vgg_fc7_features" rgb_features

  python3 make_average_video_dict.py
  python3 make_average_video_dict_flow.py

  cp -f "${DATA_DIR}/average_rgb_feats.h5" "${DATA_DIR}/average_fc7.h5"
  cp -f "${DATA_DIR}/average_flow_feats.h5" "${DATA_DIR}/average_global_flow.h5"
  popd >/dev/null
}

require_cmd wget
require_cmd tar
require_cmd unzip
require_cmd python3

download_glove

if [ -f "${DATA_DIR}/average_fc7.h5" ] && [ -f "${DATA_DIR}/average_global_flow.h5" ]; then
  echo "Found average_fc7.h5 and average_global_flow.h5. Skipping feature download."
  exit 0
fi

echo "Trying primary Berkeley feature links..."
if try_primary_average_download; then
  echo "Downloaded average features from primary links."
  exit 0
fi

echo "Primary links unavailable. Falling back to Hugging Face HTTP mirror (no torrent)."
echo "HF_RAW_DIR=${HF_RAW_DIR}"
download_hf_parts
assemble_hf_archives
build_average_features_from_hf

if [ ! -f "${DATA_DIR}/average_fc7.h5" ] || [ ! -f "${DATA_DIR}/average_global_flow.h5" ]; then
  echo "ERROR: Failed to prepare average feature files."
  exit 1
fi

echo "Prepared data/average_fc7.h5 and data/average_global_flow.h5."
echo "PyTorch checkpoints are created by build_net.py in snapshots/*.pt."
