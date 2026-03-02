from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrievalModelConfig:
    visual_feature_dim: int
    language_feature_dim: int
    sentence_length: int
    visual_embedding_dim: List[int]
    language_embedding_dim: List[int]
    vision_layers: str
    language_layers: str
    distance_function: str
    margin: float
    dropout_visual: float
    dropout_language: float
    loc: bool
    loss_type: str
    lw_inter: float
    lw_intra: float


class RetrievalNet(nn.Module):
    def __init__(self, config: RetrievalModelConfig):
        super().__init__()
        self.config = config
        self.margin = config.margin
        self.dropout_visual = nn.Dropout(p=config.dropout_visual)
        self.dropout_language = nn.Dropout(p=config.dropout_language)
        self.visual_embedding_dim = list(config.visual_embedding_dim)
        self.language_embedding_dim = list(config.language_embedding_dim)

        if config.vision_layers not in {"1", "2"}:
            raise ValueError(f"No specified vision layer for {config.vision_layers}")
        if config.language_layers != "lstm_no_embed":
            raise ValueError("Only 'lstm_no_embed' is implemented")
        if len(self.language_embedding_dim) != 2:
            raise ValueError("language_embedding_dim must have length 2")

        visual_input_dim = config.visual_feature_dim + (2 if config.loc else 0)
        if config.vision_layers == "1":
            if len(self.visual_embedding_dim) != 1:
                raise ValueError("visual_embedding_dim must have length 1 for one-layer vision model")
            self.image_embed1 = nn.Linear(visual_input_dim, self.visual_embedding_dim[0])
            self.image_embed2 = None
            visual_out_dim = self.visual_embedding_dim[0]
        else:
            if len(self.visual_embedding_dim) != 2:
                raise ValueError("visual_embedding_dim must have length 2 for two-layer vision model")
            self.image_embed1 = nn.Linear(visual_input_dim, self.visual_embedding_dim[0])
            self.image_embed2 = nn.Linear(self.visual_embedding_dim[0], self.visual_embedding_dim[1])
            visual_out_dim = self.visual_embedding_dim[1]

        self.visual_out_dim = visual_out_dim
        self.lstm = nn.LSTMCell(config.language_feature_dim, self.language_embedding_dim[0])
        self.lstm_embed = nn.Linear(self.language_embedding_dim[0], self.language_embedding_dim[1])

        self.distance_function_name = config.distance_function
        if self.distance_function_name == "eltwise_distance":
            self.eltwise_dist = nn.Linear(self.visual_out_dim, 1)
        else:
            self.eltwise_dist = None
        if self.distance_function_name == "bilinear_distance":
            bilinear_dim = self.visual_out_dim * self.visual_out_dim
            self.bilinear_dist = nn.Linear(bilinear_dim, 1)
        else:
            self.bilinear_dist = None

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.08, 0.08)
                nn.init.constant_(module.bias, 0.0)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -0.08, 0.08)
            else:
                nn.init.constant_(param, 0.0)

    def _normalize(self, x: torch.Tensor, axis: int = 1) -> torch.Tensor:
        denom = torch.sqrt(torch.sum(x * x, dim=axis, keepdim=True) + 1e-5)
        return x / denom

    def image_model(self, bottom: torch.Tensor, time_stamp: Optional[torch.Tensor]) -> torch.Tensor:
        if self.config.loc:
            if time_stamp is None:
                raise ValueError("time_stamp must be provided when loc=True")
            bottom = torch.cat([bottom, time_stamp], dim=1)
        x = self.image_embed1(bottom)
        if self.config.vision_layers == "2":
            x = F.relu(x)
            x = self.image_embed2(x)
        x = self.dropout_visual(x)
        return x

    def language_model_lstm_no_embed(self, sent_bottom: torch.Tensor, cont_bottom: torch.Tensor) -> torch.Tensor:
        # sent_bottom: [T, B, D], cont_bottom: [T, B]
        if sent_bottom.dim() != 3:
            raise ValueError("sent_bottom must be [T, B, D]")
        if cont_bottom.dim() != 2:
            raise ValueError("cont_bottom must be [T, B]")
        if sent_bottom.shape[:2] != cont_bottom.shape:
            raise ValueError("cont_bottom shape must match sent_bottom first two dims")

        t, b, _ = sent_bottom.shape
        h = sent_bottom.new_zeros((b, self.language_embedding_dim[0]))
        c = sent_bottom.new_zeros((b, self.language_embedding_dim[0]))

        for step in range(t):
            cont = cont_bottom[step].unsqueeze(1)
            h = h * cont
            c = c * cont
            h, c = self.lstm(sent_bottom[step], (h, c))

        top_text = self.lstm_embed(h)
        return top_text

    def euclidean_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        return torch.sum((vec1 - vec2) ** 2, dim=1)

    def dot_product_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sum(vec1 * vec2, dim=1)

    def eltwise_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        if self.eltwise_dist is None:
            raise RuntimeError("eltwise distance head is not initialized")
        mult = vec1 * vec2
        norm_mult = self._normalize(mult, axis=1)
        return self.eltwise_dist(norm_mult).squeeze(1)

    def bilinear_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        if self.bilinear_dist is None:
            raise RuntimeError("bilinear distance head is not initialized")
        bilinear = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).reshape(vec1.size(0), -1)
        signed = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-10)
        normalized = self._normalize(signed, axis=1)
        return self.bilinear_dist(normalized).squeeze(1)

    def distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        if self.distance_function_name == "dot_product_distance":
            return self.dot_product_distance(vec1, vec2)
        if self.distance_function_name == "eltwise_distance":
            return self.eltwise_distance(vec1, vec2)
        if self.distance_function_name == "bilinear_distance":
            return self.bilinear_distance(vec1, vec2)
        return self.euclidean_distance(vec1, vec2)

    def ranking_loss(self, positive: torch.Tensor, negative: torch.Tensor, text: torch.Tensor, lw: float) -> torch.Tensor:
        distance_p = self.distance(positive, text)
        distance_n = self.distance(negative, text)
        max_sum_margin_relu = F.relu(distance_p - distance_n + self.margin)
        return max_sum_margin_relu.mean() * lw

    def forward_embeddings(
        self,
        features_p: torch.Tensor,
        features_time_stamp_p: torch.Tensor,
        query: torch.Tensor,
        cont: torch.Tensor,
        features_inter: Optional[torch.Tensor] = None,
        features_intra: Optional[torch.Tensor] = None,
        features_time_stamp_n: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        out["embedding_visual_p"] = self.image_model(features_p, features_time_stamp_p)
        if features_inter is not None:
            out["embedding_visual_inter"] = self.image_model(features_inter, features_time_stamp_p)
        if features_intra is not None:
            out["embedding_visual_intra"] = self.image_model(features_intra, features_time_stamp_n)
        out["embedding_text"] = self.language_model_lstm_no_embed(query, cont)
        return out

    def forward_train(
        self,
        features_p: torch.Tensor,
        features_time_stamp_p: torch.Tensor,
        query: torch.Tensor,
        cont: torch.Tensor,
        features_inter: Optional[torch.Tensor] = None,
        features_intra: Optional[torch.Tensor] = None,
        features_time_stamp_n: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings = self.forward_embeddings(
            features_p=features_p,
            features_time_stamp_p=features_time_stamp_p,
            query=query,
            cont=cont,
            features_inter=features_inter,
            features_intra=features_intra,
            features_time_stamp_n=features_time_stamp_n,
        )

        text = embeddings["embedding_text"]
        losses: Dict[str, torch.Tensor] = {}
        if self.config.loss_type in {"triplet", "inter"}:
            if "embedding_visual_inter" not in embeddings:
                raise ValueError("features_inter is required for inter/triplet loss")
            losses["ranking_loss_inter"] = self.ranking_loss(
                embeddings["embedding_visual_p"],
                embeddings["embedding_visual_inter"],
                text,
                self.config.lw_inter,
            )
        if self.config.loss_type in {"triplet", "intra"}:
            if "embedding_visual_intra" not in embeddings:
                raise ValueError("features_intra is required for intra/triplet loss")
            losses["ranking_loss_intra"] = self.ranking_loss(
                embeddings["embedding_visual_p"],
                embeddings["embedding_visual_intra"],
                text,
                self.config.lw_intra,
            )
        losses["total_loss"] = sum(losses.values())
        return losses

    def score_pair(
        self,
        visual_features: torch.Tensor,
        loc_features: torch.Tensor,
        text_features: torch.Tensor,
        cont_features: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.image_model(visual_features, loc_features)
        text = self.language_model_lstm_no_embed(text_features, cont_features)
        return self.distance(visual, text)

    def get_optimizer_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        lstm_lr: float,
        language_embedding_lr: float,
    ) -> List[Dict[str, object]]:
        groups: List[Dict[str, object]] = []

        def add(params: Iterable[torch.nn.Parameter], lr_mult: float, decay_mult: float) -> None:
            params = [p for p in params if p.requires_grad]
            if not params:
                return
            groups.append(
                {
                    "params": params,
                    "lr": base_lr * lr_mult,
                    "weight_decay": weight_decay * decay_mult,
                }
            )

        add([self.image_embed1.weight], lr_mult=1.0, decay_mult=1.0)
        add([self.image_embed1.bias], lr_mult=2.0, decay_mult=0.0)
        if self.image_embed2 is not None:
            add([self.image_embed2.weight], lr_mult=1.0, decay_mult=1.0)
            add([self.image_embed2.bias], lr_mult=2.0, decay_mult=0.0)

        add(self.lstm.parameters(), lr_mult=lstm_lr, decay_mult=lstm_lr)
        add([self.lstm_embed.weight], lr_mult=language_embedding_lr, decay_mult=language_embedding_lr)
        add([self.lstm_embed.bias], lr_mult=language_embedding_lr * 2.0, decay_mult=0.0)

        if self.eltwise_dist is not None:
            add([self.eltwise_dist.weight], lr_mult=1.0, decay_mult=1.0)
            add([self.eltwise_dist.bias], lr_mult=2.0, decay_mult=0.0)
        if self.bilinear_dist is not None:
            add([self.bilinear_dist.weight], lr_mult=1.0, decay_mult=1.0)
            add([self.bilinear_dist.bias], lr_mult=2.0, decay_mult=0.0)

        return groups
