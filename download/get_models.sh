#!/bin/bash
set -euo pipefail

mkdir -p snapshots
mkdir -p prototxts
mkdir -p data

cd data

# Public Google Drive folder from the original repo README (models/data backup).
DRIVE_FOLDER_ID="1heYHAOJX0mdeLH95jxdfxry6RC_KMVyZ"
DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/${DRIVE_FOLDER_ID}"

download_or_fallback() {
  local target_name="$1"
  local primary_url="$2"
  local fallback_needed=0

  if [ -f "${target_name}" ]; then
    echo "Found ${target_name}, skipping."
    return 0
  fi

  echo "Trying primary host for ${target_name}..."
  if ! wget -O "${target_name}" "${primary_url}"; then
    rm -f "${target_name}"
    fallback_needed=1
  fi

  if [ "${fallback_needed}" -eq 1 ]; then
    echo "Primary host failed for ${target_name}. Falling back to Google Drive folder..."
    if ! command -v gdown >/dev/null 2>&1; then
      if command -v python3 >/dev/null 2>&1; then
        if ! python3 -m pip --version >/dev/null 2>&1; then
          python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
        fi
        python3 -m pip install --user --upgrade pip >/dev/null 2>&1 || true
        python3 -m pip install --user gdown >/dev/null 2>&1 || true
      fi
      if ! command -v gdown >/dev/null 2>&1 && command -v pip3 >/dev/null 2>&1; then
        pip3 install --user gdown >/dev/null 2>&1 || true
      fi
      if ! command -v gdown >/dev/null 2>&1 && command -v python >/dev/null 2>&1; then
        if ! python -m pip --version >/dev/null 2>&1; then
          python -m ensurepip --upgrade >/dev/null 2>&1 || true
        fi
        python -m pip install --user --upgrade pip >/dev/null 2>&1 || true
        python -m pip install --user gdown >/dev/null 2>&1 || true
      fi
      if ! command -v gdown >/dev/null 2>&1; then
        echo "ERROR: gdown is required for Google Drive fallback but could not be installed automatically."
        echo "Install manually and rerun:"
        echo "  python3 -m ensurepip --upgrade"
        echo "  python3 -m pip install --user gdown"
        echo "Or manually download files from:"
        echo "  ${DRIVE_FOLDER_URL}"
        exit 1
      fi
    fi
    gdown --folder "${DRIVE_FOLDER_URL}" -O .
  fi

  if [ ! -f "${target_name}" ]; then
    echo "ERROR: Could not download ${target_name}."
    echo "Open this folder and download it manually into ./data:"
    echo "${DRIVE_FOLDER_URL}"
    exit 1
  fi
}

# Pre-extracted features required for training/evaluation.
download_or_fallback "average_fc7.h5" \
  "https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5"
download_or_fallback "average_global_flow.h5" \
  "https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5"

# GloVe embedding expected by utils/data_processing.py.
if [ ! -f "glove.6B.zip" ]; then
  wget http://nlp.stanford.edu/data/glove.6B.zip
fi
unzip -o glove.6B.zip
rm -f glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt

cd ..

echo "Downloaded features and embeddings."
echo "PyTorch checkpoints are created by build_net.py (no Caffe .caffemodel files are used)."
