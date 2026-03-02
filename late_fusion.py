from __future__ import annotations

import argparse
import pickle as pkl
import sys

import numpy as np

sys.path.append("utils/")
from config import result_dir  # noqa: E402
from data_processing import possible_segments  # noqa: E402
from eval import eval_predictions  # noqa: E402
from utils import read_json  # noqa: E402


def late_fusion(rgb_tag, flow_tag, split, iteration, lambda_values):
    data = read_json(f"data/{split}_data.json")

    # Read in raw scores from RGB/flow models.
    with open(f"{result_dir}/{rgb_tag}_{split}.p", "rb") as handle:
        rgb_results = pkl.load(handle)
    with open(f"{result_dir}/{flow_tag}_{split}.p", "rb") as handle:
        flow_results = pkl.load(handle)

    for lambda_value in lambda_values:
        all_segments = []
        print(f"Lambda {lambda_value:.6f}:")
        for datum in data:
            rgb_scores = rgb_results[iteration][datum["annotation_id"]]
            flow_scores = flow_results[iteration][datum["annotation_id"]]
            scores = lambda_value * rgb_scores + (1 - lambda_value) * flow_scores
            all_segments.append([possible_segments[i] for i in np.argsort(scores)])
        eval_predictions(all_segments, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb_tag", type=str, default=None)
    parser.add_argument("--flow_tag", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--iter", type=int, default=30000)
    parser.add_argument(
        "--lambda_values",
        type=float,
        nargs="+",
        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    args = parser.parse_args()
    late_fusion(args.rgb_tag, args.flow_tag, args.split, args.iter, args.lambda_values)
