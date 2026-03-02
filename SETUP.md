# SETUP

This guide is for the PyTorch version of `LocalizingMoments`.
It uses only HTTP downloads (no torrent flow).

## 1) Environment (WSL/Linux recommended)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you do not need CUDA, you can install a CPU-only PyTorch wheel first, then install the rest.
- Keep very large data files on Linux filesystem (for example under `~/didemo_cache`) instead of `/mnt/d`.

## 2) Data Setup (non-torrent)

### Option A: One command (recommended)

```bash
bash download/get_models.sh
```

What this does:
- downloads GloVe embeddings
- tries original Berkeley average-feature links
- if Berkeley returns 403, downloads feature archives from Hugging Face over HTTP
- builds:
  - `data/average_fc7.h5`
  - `data/average_global_flow.h5`

The script stores heavy raw archives in `${HF_RAW_DIR}`.
Default: `~/didemo_cache/hf_raw`

You can override:

```bash
HF_RAW_DIR=~/my_didemo_cache/hf_raw bash download/get_models.sh
```

### Option B: Manual HTTP download (if you want full control)

```bash
mkdir -p ~/didemo_cache/hf_raw
cd ~/didemo_cache/hf_raw

for i in 0 1 2 3; do
  wget -c "https://huggingface.co/datasets/YimuWang/didemon_retrieval/resolve/main/flow_features.tar.$i"
done

for i in $(seq -w 0 12); do
  wget -c "https://huggingface.co/datasets/YimuWang/didemon_retrieval/resolve/main/rgb_vgg_fc7_features.tar.$i"
done

cat flow_features.tar.0 flow_features.tar.1 flow_features.tar.2 flow_features.tar.3 > flow_features.tar
cat rgb_vgg_fc7_features.tar.{00,01,02,03,04,05,06,07,08,09,10,11,12} > rgb_vgg_fc7_features.tar

tar -tf flow_features.tar > /dev/null
tar -tf rgb_vgg_fc7_features.tar > /dev/null

tar -xf flow_features.tar
tar -xf rgb_vgg_fc7_features.tar
```

Then link and build averages:

```bash
cd /path/to/LocalizingMoments
source .venv/bin/activate

ln -sfn ~/didemo_cache/hf_raw/flow_features flow_features
ln -sfn ~/didemo_cache/hf_raw/rgb_vgg_fc7_features rgb_features

python3 make_average_video_dict.py
python3 make_average_video_dict_flow.py

cp data/average_rgb_feats.h5 data/average_fc7.h5
cp data/average_flow_feats.h5 data/average_global_flow.h5
```

## 3) Verify required files

From repo root:

```bash
ls -lh data/average_fc7.h5 data/average_global_flow.h5 data/glove.6B.300d.txt
```

## 4) Train

RGB:

```bash
bash run_job_rgb.sh
```

Flow:

```bash
bash run_job_flow.sh
```

## 5) Evaluate

```bash
bash test_network.sh <rgb_snapshot_tag> <flow_snapshot_tag> [iter]
```

Example:

```bash
bash test_network.sh rgb_hachiko_ flow_hachiko_ 30000
```

## Troubleshooting

### Berkeley download returns 403
That is expected now for many users. Use `download/get_models.sh` (HF fallback) or Option B above.

### `Input/output error` while `cat` or `tar`
Usually filesystem instability on `/mnt/d` with huge files. Move raw chunks to Linux filesystem:

```bash
mkdir -p ~/didemo_cache/hf_raw
```

Redownload there and retry concatenate/extract.

### `tar: Unexpected EOF in archive`
One or more chunks are incomplete/corrupt. Re-download missing chunk(s) with `wget -c` and re-run `cat`.

### `OSError ... truncated file` when running `make_average_video_dict.py`
At least one `.h5` file in `rgb_features/` is corrupt. Rebuild `rgb_vgg_fc7_features.tar` from complete chunks, re-extract, then rerun.

### `UnpicklingError ... STRING opcode argument must be quoted`
Update to this latest repo state and rerun. `make_average_video_dict_flow.py` now handles CRLF-corrupted `data/frame_rate_clean.p` safely.

## References

- Hugging Face mirror: https://huggingface.co/datasets/YimuWang/didemon_retrieval
