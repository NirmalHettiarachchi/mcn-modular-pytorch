# Localizing Moments in Video with Natural Language (PyTorch)

Hendricks, Lisa Anne, et al. "Localizing Moments in Video with Natural Language." ICCV (2017).

Paper: https://arxiv.org/pdf/1708.01641.pdf  
Project page: https://people.eecs.berkeley.edu/~lisa_anne/didemo.html

```bibtex
@inproceedings{hendricks17iccv,
  title = {Localizing Moments in Video with Natural Language.},
  author = {Hendricks, Lisa Anne and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2017}
}
```

License: BSD 2-Clause

## Quick Start

This repository is fully migrated to PyTorch (no Caffe runtime).

Use the full setup guide in `SETUP.md`.
It includes:
- venv setup
- non-torrent data setup (HTTP downloads only)
- building `data/average_fc7.h5` and `data/average_global_flow.h5`
- training and evaluation commands

If you already have a working environment and only need data:

```bash
bash download/get_models.sh
```

`download/get_models.sh` now uses:
1. Original Berkeley links (if available)
2. Hugging Face HTTP mirror fallback (no torrent)

## Training

Train RGB model:

```bash
bash run_job_rgb.sh
```

Train flow model:

```bash
bash run_job_flow.sh
```

Checkpoints are saved as:

```text
snapshots/<snapshot_tag>_iter_<iter>.pt
```

Deploy metadata is saved as:

```text
prototxts/deploy_clip_retrieval_<snapshot_tag>.json
```

## Evaluation

Run RGB + flow + late fusion on val/test:

```bash
bash test_network.sh <rgb_snapshot_tag> <flow_snapshot_tag> [iter]
```

Example:

```bash
bash test_network.sh rgb_hachiko_ flow_hachiko_ 30000
```

Evaluation utilities are in `utils/eval.py`.

## Dataset Notes

Annotations are in the `data` folder and contain temporally grounded descriptions.
Videos are split into 5-second chunks during annotation.

Key fields:
- `annotation_id`
- `description`
- `video`
- `times`
- `download_link`
- `num_segments`

## Feature Sources

The original Berkeley-hosted average features can return HTTP 403.
For reliable setup, use `SETUP.md` or `download/get_models.sh`, which fall back to the public Hugging Face dataset mirror:

- https://huggingface.co/datasets/YimuWang/didemon_retrieval

No torrent workflow is required.
