#!/bin/bash
set -euo pipefail

mkdir -p snapshots
mkdir -p prototxts
mkdir -p data

cd data

# Public Google Drive folder from the original repo README (models/data backup).
DRIVE_FOLDER_ID="1heYHAOJX0mdeLH95jxdfxry6RC_KMVyZ"
DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/${DRIVE_FOLDER_ID}"
TORRENT_URL="https://orion.hyper.ai/tracker/download?torrent=21327"
TORRENT_FILE="didemo.torrent"
TORRENT_ROOT_DIR="DiDeMo"

download_or_fallback() {
  local target_name="$1"
  local primary_url="$2"
  local torrent_src_name="$3"
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
    if ! gdown --folder "${DRIVE_FOLDER_URL}" -O .; then
      echo "Google Drive folder download failed for ${target_name}."
      echo "This usually means the folder is not publicly listable or is quota-limited."

      echo "Trying public torrent mirror fallback (no Google credentials needed)..."
      if ! command -v aria2c >/dev/null 2>&1; then
        echo "aria2c is required for torrent fallback."
        echo "Install and rerun:"
        echo "  sudo apt update && sudo apt install -y aria2"
      else
        if [ ! -f "${TORRENT_FILE}" ]; then
          wget -O "${TORRENT_FILE}" "${TORRENT_URL}"
        fi
        # Files 3 and 4 in this torrent are:
        # 3 -> data/average_flow_feats.h5, 4 -> data/average_rgb_feats.h5
        aria2c --seed-time=0 --select-file=3,4 --dir . "${TORRENT_FILE}" || true
      fi
    fi

    # Map mirrored filenames to expected local filenames.
    if [ -f "${TORRENT_ROOT_DIR}/data/${torrent_src_name}" ] && [ ! -f "${target_name}" ]; then
      cp "${TORRENT_ROOT_DIR}/data/${torrent_src_name}" "${target_name}"
    fi
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
  "https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5" \
  "average_rgb_feats.h5"
download_or_fallback "average_global_flow.h5" \
  "https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5" \
  "average_flow_feats.h5"

# GloVe embedding expected by utils/data_processing.py.
if [ ! -f "glove.6B.zip" ]; then
  wget http://nlp.stanford.edu/data/glove.6B.zip
fi
unzip -o glove.6B.zip
rm -f glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt

cd ..

echo "Downloaded features and embeddings."
echo "PyTorch checkpoints are created by build_net.py (no Caffe .caffemodel files are used)."
