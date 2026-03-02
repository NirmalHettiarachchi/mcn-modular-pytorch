#!/bin/bash
set -euo pipefail

mkdir -p snapshots
mkdir -p prototxts
mkdir -p data

cd data

# Pre-extracted features required for training/evaluation.
wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5
wget https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5

# GloVe embedding expected by utils/data_processing.py.
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -o glove.6B.zip
rm -f glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt

cd ..

echo "Downloaded features and embeddings."
echo "PyTorch checkpoints are created by build_net.py (no Caffe .caffemodel files are used)."
