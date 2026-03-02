from __future__ import annotations

import argparse
import json
import os
import pickle as pkl
import sys
from typing import Dict

import numpy as np
import torch

from pytorch_model import RetrievalModelConfig, RetrievalNet

sys.path.append("utils/")
from config import result_dir, snapshot_dir  # noqa: E402
from data_processing import (  # noqa: E402
    extractLanguageFeatures,
    extractVisualFeatures,
    language_feature_process_dict,
    possible_segments,
)
from eval import eval_predictions  # noqa: E402
from utils import read_json  # noqa: E402


def load_model_for_iteration(
    checkpoint_path: str,
    default_model_config: Dict[str, object],
    device: torch.device,
) -> RetrievalNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        model_config_dict = checkpoint["model_config"]
    else:
        model_config_dict = default_model_config

    model_config = RetrievalModelConfig(**model_config_dict)
    model = RetrievalNet(model_config).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def checkpoint_path_for_iter(deploy_net: str, snapshot_tag: str, iteration: int) -> str:
    if deploy_net and os.path.exists(deploy_net):
        try:
            with open(deploy_net, "r", encoding="utf-8") as handle:
                deploy_cfg = json.load(handle)
            pattern = deploy_cfg.get("checkpoint_pattern")
            if pattern:
                return pattern.format(iter=iteration)
        except Exception:
            pass
    return os.path.join(snapshot_dir, f"{snapshot_tag}_iter_{iteration}.pt")


def default_model_config_from_args(
    visual_feature_dim: int,
    language_feature_dim: int,
    visual_embedding_dim,
    language_embedding_dim,
    vision_layers: str,
    language_layers: str,
    distance_function: str,
    margin: float,
    dropout_visual: float,
    dropout_language: float,
    loc: bool,
):
    return {
        "visual_feature_dim": visual_feature_dim,
        "language_feature_dim": language_feature_dim,
        "sentence_length": 50,
        "visual_embedding_dim": list(visual_embedding_dim),
        "language_embedding_dim": list(language_embedding_dim),
        "vision_layers": vision_layers,
        "language_layers": language_layers,
        "distance_function": distance_function,
        "margin": margin,
        "dropout_visual": dropout_visual,
        "dropout_language": dropout_language,
        "loc": loc,
        "loss_type": "triplet",
        "lw_inter": 0.5,
        "lw_intra": 0.5,
    }


def test_model(
    deploy_net,
    snapshot_tag,
    visual_feature="feature_process_norm",
    language_feature="recurrent_embedding",
    max_iter=30000,
    snapshot_interval=30000,
    loc=False,
    test_h5="data/average_fc7.h5",
    split="val",
    gpu=0,
    visual_embedding_dim=(500, 100),
    language_embedding_dim=(1000, 100),
    vision_layers="2",
    language_layers="lstm_no_embed",
    distance_function="euclidean_distance",
    margin=0.1,
    dropout_visual=0.0,
    dropout_language=0.0,
):
    params = {
        "feature_process": visual_feature,
        "loc_feature": loc,
        "loss_type": "triplet",
        "batch_size": 120,
        "features": test_h5,
        "oversample": False,
        "sentence_length": 50,
        "query_key": "query",
        "cont_key": "cont",
        "feature_key_p": "features_p",
        "feature_time_stamp_p": "feature_time_stamp_p",
        "feature_time_stamp_n": "feature_time_stamp_n",
    }

    language_extractor_fcn = extractLanguageFeatures
    visual_extractor_fcn = extractVisualFeatures

    language_process = language_feature_process_dict[language_feature]
    data_orig = read_json(f"data/{split}_data.json")
    language_processor = language_process(data_orig)
    data = language_processor.preprocess(data_orig)
    params["vocab_dict"] = language_processor.vocab_dict
    num_glove_centroids = language_processor.get_vector_dim()
    params["num_glove_centroids"] = num_glove_centroids
    thread_result = {}

    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments_local = visual_feature_extractor.possible_annotations

    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    default_model_config = default_model_config_from_args(
        visual_feature_dim=visual_feature_extractor.feature_dim,
        language_feature_dim=num_glove_centroids,
        visual_embedding_dim=visual_embedding_dim,
        language_embedding_dim=language_embedding_dim,
        vision_layers=vision_layers,
        language_layers=language_layers,
        distance_function=distance_function,
        margin=margin,
        dropout_visual=dropout_visual,
        dropout_language=dropout_language,
        loc=loc,
    )

    all_scores = {}
    for iteration in range(snapshot_interval, max_iter + 1, snapshot_interval):
        sorted_segments_list = []
        checkpoint_path = checkpoint_path_for_iter(deploy_net, snapshot_tag, iteration)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = load_model_for_iteration(checkpoint_path, default_model_config, device)
        all_scores[iteration] = {}

        # Determine score for segments in each video.
        with torch.no_grad():
            for idx, datum in enumerate(data):
                vis_features, loc_features = visual_feature_extractor.get_data_test({"video": datum["video"]})
                lang_features, cont = textual_feature_extractor.get_data_test(datum)

                n_segments = vis_features.shape[0]
                text_batch = np.repeat(lang_features[:, np.newaxis, :], n_segments, axis=1)
                cont_batch = np.repeat(cont[:, np.newaxis], n_segments, axis=1)

                vis_tensor = torch.from_numpy(vis_features).to(device)
                loc_tensor = torch.from_numpy(loc_features).to(device)
                text_tensor = torch.from_numpy(text_batch).to(device)
                cont_tensor = torch.from_numpy(cont_batch).to(device)

                scores = model.score_pair(vis_tensor, loc_tensor, text_tensor, cont_tensor)
                scores_np = scores.detach().cpu().numpy().squeeze()

                sorted_segments = [possible_segments_local[i] for i in np.argsort(scores_np)]
                sorted_segments_list.append(sorted_segments)
                all_scores[iteration][datum["annotation_id"]] = scores_np.copy()

                if idx % 10 == 0:
                    sys.stdout.write(f"\r{idx}/{len(data)}")

        eval_predictions(sorted_segments_list, data)

    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/{snapshot_tag}_{split}.p"
    with open(output_path, "wb") as handle:
        pkl.dump(all_scores, handle)
    print(f"Dumped results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--deploy_net", type=str, default=None)
    parser.add_argument("--snapshot_tag", type=str, default=None)
    parser.add_argument("--visual_feature", type=str, default="feature_process_norm")
    parser.add_argument("--language_feature", type=str, default="recurrent_embedding")
    parser.add_argument("--max_iter", type=int, default=30000)
    parser.add_argument("--snapshot_interval", type=int, default=30000)
    parser.add_argument("--loc", dest="loc", action="store_true")
    parser.set_defaults(loc=False)
    parser.add_argument("--test_h5", type=str, default="data/average_fc7.h5")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--gpu", type=int, default=0)

    # Optional overrides when no deploy config is provided.
    parser.add_argument("--visual_embedding_dim", type=int, nargs="+", default=[500, 100])
    parser.add_argument("--language_embedding_dim", type=int, nargs="+", default=[1000, 100])
    parser.add_argument("--vision_layers", type=str, default="2")
    parser.add_argument("--language_layers", type=str, default="lstm_no_embed")
    parser.add_argument("--distance_function", type=str, default="euclidean_distance")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--dropout_visual", type=float, default=0.0)
    parser.add_argument("--dropout_language", type=float, default=0.0)

    args = parser.parse_args()

    test_model(
        args.deploy_net,
        args.snapshot_tag,
        visual_feature=args.visual_feature,
        language_feature=args.language_feature,
        max_iter=args.max_iter,
        snapshot_interval=args.snapshot_interval,
        loc=args.loc,
        test_h5=args.test_h5,
        split=args.split,
        gpu=args.gpu,
        visual_embedding_dim=args.visual_embedding_dim,
        language_embedding_dim=args.language_embedding_dim,
        vision_layers=args.vision_layers,
        language_layers=args.language_layers,
        distance_function=args.distance_function,
        margin=args.margin,
        dropout_visual=args.dropout_visual,
        dropout_language=args.dropout_language,
    )
