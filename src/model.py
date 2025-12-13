from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from loss import ArcMarginProduct


class Backbone(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone_name == "resnet18":
            net = models.resnet18(pretrained=pretrained)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone_name == "resnet50":
            net = models.resnet50(pretrained=pretrained)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.net = net
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReIDNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
    ):
        super().__init__()
        self.backbone = Backbone(backbone_name=backbone_name, pretrained=pretrained)
        self.embedding = nn.Linear(self.backbone.out_dim, embedding_dim)
        self.arc_head = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=arcface_s,
            m=arcface_m,
        )

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.backbone(x)  # [B, feat_dim]
        emb = self.embedding(feat)  # [B, D]
        emb = F.normalize(emb, dim=1)

        if labels is not None:
            logits = self.arc_head(emb, labels)
            return emb, logits
        return emb, None
