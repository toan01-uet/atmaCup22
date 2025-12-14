from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    ArcFace head: https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # input: [B, in_features], label: [B]
        normalized_input = F.normalize(input)
        normalized_weight = F.normalize(self.weight)

        cosine = F.linear(normalized_input, normalized_weight)  # [B, C]
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)

        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.m)

        one_hot = F.one_hot(label, num_classes=self.out_features).float()
        logits = cosine * (1.0 - one_hot) + target_logit * one_hot
        logits *= self.s

        return logits


class TripletLoss(nn.Module):
    """
    Triplet loss đơn giản với mining trong batch (semi-hard / all-pairs).
    Ở đây dùng batch-all đơn giản cho pseudo.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, D]
        labels: [B]
        """
        device = embeddings.device
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        labels_not_equal = ~labels_equal

        mask_anchor_positive = labels_equal ^ torch.eye(
            labels.size(0), dtype=torch.bool, device=device
        )
        mask_anchor_negative = labels_not_equal

        # hardest positive
        dist_ap = (pairwise_dist * mask_anchor_positive.float()).max(dim=1)[0]
        # hardest negative
        max_dist = pairwise_dist.max().detach()
        dist_an = pairwise_dist + max_dist * (~mask_anchor_negative).float()
        dist_an = dist_an.min(dim=1)[0]

        losses = F.relu(dist_ap - dist_an + self.margin)
        return losses.mean()
