#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <rgb_snapshot_tag> <flow_snapshot_tag> [iter]"
  exit 1
fi

RGB_TAG="$1"
FLOW_TAG="$2"
ITER="${3:-30000}"

echo "RGB model on val..."
"${PYTHON_BIN}" test_network.py --deploy_net "prototxts/deploy_clip_retrieval_${RGB_TAG}.json" \
                       --snapshot_tag "${RGB_TAG}" \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter "${ITER}" \
                       --snapshot_interval "${ITER}" \
                       --loc \
                       --test_h5 data/average_fc7.h5 \
                       --split val

echo "Flow model on val..."
"${PYTHON_BIN}" test_network.py --deploy_net "prototxts/deploy_clip_retrieval_${FLOW_TAG}.json" \
                       --snapshot_tag "${FLOW_TAG}" \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter "${ITER}" \
                       --snapshot_interval "${ITER}" \
                       --loc \
                       --test_h5 data/average_global_flow.h5 \
                       --split val

echo "Fusion model on val..."
"${PYTHON_BIN}" late_fusion.py --rgb_tag "${RGB_TAG}" \
                      --flow_tag "${FLOW_TAG}" \
                      --split val \
                      --iter "${ITER}"

echo "RGB model on test..."
"${PYTHON_BIN}" test_network.py --deploy_net "prototxts/deploy_clip_retrieval_${RGB_TAG}.json" \
                       --snapshot_tag "${RGB_TAG}" \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter "${ITER}" \
                       --snapshot_interval "${ITER}" \
                       --loc \
                       --test_h5 data/average_fc7.h5 \
                       --split test

echo "Flow model on test..."
"${PYTHON_BIN}" test_network.py --deploy_net "prototxts/deploy_clip_retrieval_${FLOW_TAG}.json" \
                       --snapshot_tag "${FLOW_TAG}" \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter "${ITER}" \
                       --snapshot_interval "${ITER}" \
                       --loc \
                       --test_h5 data/average_global_flow.h5 \
                       --split test

echo "Fusion model on test..."
"${PYTHON_BIN}" late_fusion.py --rgb_tag "${RGB_TAG}" \
                      --flow_tag "${FLOW_TAG}" \
                      --split test \
                      --iter "${ITER}" \
                      --lambda_values 0.5
