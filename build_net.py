from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict

import h5py
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from pytorch_model import RetrievalModelConfig, RetrievalNet

import sys

sys.path.append("utils/")
from config import result_dir, snapshot_dir  # noqa: E402
from data_processing import (  # noqa: E402
    batchAdvancer,
    build_preprocessed_data,
    extractLanguageFeatures,
    extractVisualFeatures,
    feature_process_dict,
)
from utils import read_json  # noqa: E402


def add_dict_values(key, my_dict):
    if my_dict.values():
        max_value = max(my_dict.values())
        my_dict[key] = max_value + 1
    else:
        my_dict[key] = 0
    return my_dict


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_visual_feature_dim(features_path: str, feature_process_name: str) -> int:
    with h5py.File(features_path, "r") as handle:
        first_key = next(iter(handle.keys()))
        feat = np.array(handle[first_key])
    processed = feature_process_dict[feature_process_name](0, 0, feat)
    return int(processed.shape[-1])


def normalize_solver_type(solver_type: str) -> str:
    return solver_type.replace('"', "").strip().lower()


def build_checkpoint_payload(
    model: RetrievalNet,
    model_config: RetrievalModelConfig,
    args: argparse.Namespace,
    tag: str,
    iteration: int,
    visual_feature_dim: int,
    language_feature_dim: int,
) -> Dict[str, object]:
    return {
        "tag": tag,
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "args": vars(args),
        "visual_feature_dim": visual_feature_dim,
        "language_feature_dim": language_feature_dim,
    }


def write_json(path: str, payload: Dict[str, object]) -> None:
    def _json_sanitize(value: Any):
        if isinstance(value, dict):
            return {str(k): _json_sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_sanitize(v) for v in value]
        if isinstance(value, np.ndarray):
            return {"ndarray_shape": list(value.shape)}
        if isinstance(value, np.generic):
            return value.item()
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_sanitize(payload), handle, indent=2, sort_keys=True)
    print(f"Wrote config to: {path}")


def maybe_load_pretrained(model: RetrievalNet, pretrained_model: str, device: torch.device) -> None:
    if not pretrained_model:
        return
    checkpoint = torch.load(pretrained_model, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    print(f"Copying weights from {pretrained_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # how to tag built snapshots and metadata
    parser.add_argument("--tag", type=str, default="")

    # training data
    parser.add_argument("--train_json", type=str, default="data/train_data.json")
    parser.add_argument("--train_h5", type=str, default="data/average_fc7.h5")
    parser.add_argument("--test_json", type=str, default="data/val_data.json")
    parser.add_argument("--test_h5", type=str, default="data/average_fc7.h5")

    # net specifications
    parser.add_argument("--feature_process_visual", type=str, default="feature_process_norm")
    parser.add_argument("--feature_process_language", type=str, default="recurrent_embedding")
    parser.add_argument("--loc", dest="loc", action="store_true")
    parser.add_argument("--no-loc", dest="loc", action="store_false")
    parser.set_defaults(loc=False)
    parser.add_argument("--loss_type", type=str, default="triplet")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--dropout_visual", type=float, default=0.0)
    parser.add_argument("--dropout_language", type=float, default=0.0)
    parser.add_argument("--visual_embedding_dim", type=int, nargs="+", default=[100])
    parser.add_argument("--language_embedding_dim", type=int, nargs="+", default=[1000, 100])
    parser.add_argument("--lw_inter", type=float, default=0.5)
    parser.add_argument("--lw_intra", type=float, default=0.5)
    parser.add_argument("--vision_layers", type=str, default="1")
    parser.add_argument("--language_layers", type=str, default="lstm_no_embed")
    parser.add_argument("--distance_function", type=str, default="euclidean_distance")
    parser.add_argument("--image_tag", type=str, default=None)

    # learning params
    parser.add_argument("--random_seed", type=int, default=1701)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--snapshot", type=int, default=5000)
    parser.add_argument("--stepsize", type=int, default=5000)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--lstm_lr", type=float, default=10)
    parser.add_argument("--language_embedding_lr", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--solver_type", type=str, default='"SGD"')
    parser.add_argument("--delta", type=float, default=1e-8)  # only for Adam
    args = parser.parse_args()

    print(f"Feature process visual: {args.feature_process_visual}")
    print(f"Feature process language: {args.feature_process_language}")
    print(f"Loc: {args.loc}")
    print(f"Dropout visual {args.dropout_visual:.6f}")
    print(f"Dropout language {args.dropout_language:.6f}")
    print(f"Pretrained model {args.pretrained_model}")

    valid_loss_type = ["triplet", "inter", "intra"]
    if args.loss_type not in valid_loss_type:
        raise ValueError(f"loss_type must be one of {valid_loss_type}")
    if args.lw_inter < 0 or args.lw_intra < 0:
        raise ValueError("loss weights must be >= 0")
    if args.loss_type == "inter":
        args.lw_inter = 1
        args.lw_intra = 0
    if args.loss_type == "intra":
        args.lw_intra = 1
        args.lw_inter = 0

    os.makedirs("prototxts", exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    set_random_seed(args.random_seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"Using device: {device}")

    train_base = "prototxts/train_clip_retrieval_%s.json"
    deploy_base = "prototxts/deploy_clip_retrieval_%s.json"
    snapshot_base = "clip_retrieval_"

    params = {}
    params["sentence_length"] = 50
    params["descriptions"] = args.train_json
    params["features"] = args.train_h5
    params["top_names"] = ["features_p", "query", "features_time_stamp_p", "features_time_stamp_n"]
    params["top_names_dict"] = {}
    for key in params["top_names"]:
        params["top_names_dict"] = add_dict_values(key, params["top_names_dict"])
    params["feature_process"] = args.feature_process_visual
    params["loc_feature"] = args.loc
    params["language_feature"] = args.feature_process_language
    params["loss_type"] = args.loss_type
    params["batch_size"] = args.batch_size

    if args.loss_type in ["triplet", "inter"]:
        inter_top_name = "features_inter"
        params["top_names"].append(inter_top_name)
        params["top_names_dict"] = add_dict_values(inter_top_name, params["top_names_dict"])
    if args.loss_type in ["triplet", "intra"]:
        intra_top_name = "features_intra"
        params["top_names"].append(intra_top_name)
        params["top_names_dict"] = add_dict_values(intra_top_name, params["top_names_dict"])

    if args.language_layers in ["lstm", "lstm_no_embed", "gru", "gru_no_embed"]:
        params["top_names"].append("cont")
        params["top_names_dict"] = add_dict_values("cont", params["top_names_dict"])
        params["sentence_length"] = 50

    top_size = len(params["top_names"])
    _ = top_size  # kept for parity with original setup

    visual_feature_dim = get_visual_feature_dim(params["features"], args.feature_process_visual)

    language_processor_for_dim = build_preprocessed_data(
        params["descriptions"], params["language_feature"]
    )[1]
    language_feature_dim = language_processor_for_dim.get_vector_dim()
    vocab_size = language_processor_for_dim.get_vocab_size()
    params["vocab_size"] = vocab_size

    pretrained_model_bool = bool(args.pretrained_model)
    tag = (
        f"{snapshot_base}{args.tag}{args.feature_process_visual}_{args.feature_process_language}_"
        f"lf{str(args.loc)}_dv{str(args.dropout_visual)}_dl{str(args.dropout_language)}_"
        f"nlv{args.vision_layers}_nll{args.language_layers}_"
        f"edl{'-'.join(str(a) for a in args.language_embedding_dim)}_"
        f"edv{'-'.join(str(a) for a in args.visual_embedding_dim)}_"
        f"pm{str(pretrained_model_bool)}_loss{args.loss_type}_lwInter{str(args.lw_inter)}"
    )

    train_path = train_base % tag
    deploy_path = deploy_base % tag

    train_data, language_processor = build_preprocessed_data(
        params["descriptions"], params["language_feature"]
    )
    params["vocab_dict"] = language_processor.vocab_dict
    params["num_glove_centroids"] = language_processor.get_vector_dim()
    params["query_key"] = "query"
    params["feature_key_n"] = "features_n"
    params["feature_key_p"] = "features_p"
    params["feature_key_t"] = "features_t"
    params["feature_time_stamp_p"] = "features_time_stamp_p"
    params["feature_time_stamp_n"] = "features_time_stamp_n"
    params["cont_key"] = "cont"

    thread_result: Dict[str, np.ndarray] = {}
    visual_feature_extractor = extractVisualFeatures(train_data, params, thread_result)
    textual_feature_extractor = extractLanguageFeatures(train_data, params, thread_result)
    batch_advancer = batchAdvancer([visual_feature_extractor, textual_feature_extractor])

    model_config = RetrievalModelConfig(
        visual_feature_dim=visual_feature_dim,
        language_feature_dim=language_feature_dim,
        sentence_length=params["sentence_length"],
        visual_embedding_dim=args.visual_embedding_dim,
        language_embedding_dim=args.language_embedding_dim,
        vision_layers=args.vision_layers,
        language_layers=args.language_layers,
        distance_function=args.distance_function,
        margin=args.margin,
        dropout_visual=args.dropout_visual,
        dropout_language=args.dropout_language,
        loc=args.loc,
        loss_type=args.loss_type,
        lw_inter=args.lw_inter,
        lw_intra=args.lw_intra,
    )

    model = RetrievalNet(model_config).to(device)
    maybe_load_pretrained(model, args.pretrained_model, device)

    optimizer_groups = model.get_optimizer_param_groups(
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        lstm_lr=args.lstm_lr,
        language_embedding_lr=args.language_embedding_lr,
    )
    solver_type = normalize_solver_type(args.solver_type)
    if solver_type == "adam":
        optimizer = torch.optim.Adam(optimizer_groups, eps=args.delta, betas=(0.9, 0.999))
        scheduler = None
    else:
        optimizer = torch.optim.SGD(optimizer_groups, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.stepsize), gamma=0.1
        )

    iter_size = max(1, 120 // max(1, args.batch_size))
    print(f"Gradient accumulation steps (iter_size): {iter_size}")

    write_json(
        train_path,
        {
            "tag": tag,
            "mode": "train",
            "params": params,
            "args": vars(args),
            "model_config": asdict(model_config),
        },
    )
    write_json(
        deploy_path,
        {
            "tag": tag,
            "mode": "deploy",
            "sentence_length": params["sentence_length"],
            "visual_feature_dim": visual_feature_dim,
            "language_feature_dim": language_feature_dim,
            "model_config": asdict(model_config),
            "checkpoint_pattern": os.path.join(snapshot_dir, f"{tag}_iter_{{iter}}.pt"),
        },
    )

    model.train()
    for iteration in range(1, args.max_iter + 1):
        optimizer.zero_grad(set_to_none=True)
        summed_losses: Dict[str, float] = {}

        for _ in range(iter_size):
            thread_result.clear()
            batch_advancer()

            features_p = torch.from_numpy(thread_result["features_p"]).to(device)
            query = torch.from_numpy(thread_result["query"]).to(device)
            cont = torch.from_numpy(thread_result["cont"]).to(device)
            features_time_stamp_p = torch.from_numpy(thread_result["features_time_stamp_p"]).to(device)
            features_time_stamp_n = torch.from_numpy(thread_result["features_time_stamp_n"]).to(device)

            features_inter = (
                torch.from_numpy(thread_result["features_inter"]).to(device)
                if args.loss_type in {"triplet", "inter"}
                else None
            )
            features_intra = (
                torch.from_numpy(thread_result["features_intra"]).to(device)
                if args.loss_type in {"triplet", "intra"}
                else None
            )

            losses = model.forward_train(
                features_p=features_p,
                features_time_stamp_p=features_time_stamp_p,
                query=query,
                cont=cont,
                features_inter=features_inter,
                features_intra=features_intra,
                features_time_stamp_n=features_time_stamp_n,
            )
            total_loss = losses["total_loss"] / float(iter_size)
            total_loss.backward()

            for name, value in losses.items():
                summed_losses[name] = summed_losses.get(name, 0.0) + float(value.item()) / float(
                    iter_size
                )

        clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if iteration % 10 == 0 or iteration == 1:
            inter_txt = f"{summed_losses.get('ranking_loss_inter', 0.0):.6f}"
            intra_txt = f"{summed_losses.get('ranking_loss_intra', 0.0):.6f}"
            total_txt = f"{summed_losses.get('total_loss', 0.0):.6f}"
            print(
                f"[Iter {iteration}/{args.max_iter}] total_loss={total_txt} "
                f"inter={inter_txt} intra={intra_txt}"
            )

        if iteration % args.snapshot == 0 or iteration == args.max_iter:
            snapshot_path = os.path.join(snapshot_dir, f"{tag}_iter_{iteration}.pt")
            checkpoint_payload = build_checkpoint_payload(
                model=model,
                model_config=model_config,
                args=args,
                tag=tag,
                iteration=iteration,
                visual_feature_dim=visual_feature_dim,
                language_feature_dim=language_feature_dim,
            )
            torch.save(checkpoint_payload, snapshot_path)
            print(f"Wrote snapshot to: {snapshot_path}")
