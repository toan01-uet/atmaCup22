from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
from src.models.model import ReIDNet
from src.utils.loss import TripletLoss
from src.utils.metric import macro_f1

class PlayerReIDModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        embedding_dim: int = 512,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
        lr: float = 1e-4, #hparams
        weight_decay: float = 1e-5, #hparams
        lambda_triplet: float = 1.0, #hparams
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ReIDNet(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            backbone_name=backbone_name,
            pretrained=True,
            arcface_s=arcface_s,
            arcface_m=arcface_m,
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3)
        
        self.lambda_triplet = lambda_triplet
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(x, labels)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        imgs, labels = batch
        embeddings, logits = self(imgs, labels)

        loss_id = self.ce_loss(logits, labels)
        loss_tri = self.triplet_loss(embeddings, labels)
        loss = loss_id + self.lambda_triplet * loss_tri

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_id", loss_id, on_step=True, on_epoch=True)
        self.log("train/loss_tri", loss_tri, on_step=True, on_epoch=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # Log to wandb every 10 steps
        if batch_idx % 10 == 0:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/loss_id_step": loss_id.item(),
                "train/loss_tri_step": loss_tri.item(),
                "train/acc_step": acc.item(),
                "global_step": self.global_step,
            })

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        imgs, labels = batch
        embeddings, logits = self(imgs, labels)
        loss_id = self.ce_loss(logits, labels)
        # Skip triplet loss in validation to avoid issues with small batches
        loss = loss_id

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        f1 = macro_f1(preds, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", f1, on_epoch=True, prog_bar=True)

        # Log to wandb every 10 steps
        if batch_idx % 10 == 0:
            wandb.log({
                "val/loss_step": loss.item(),
                "val/acc_step": acc.item(),
                "val/f1_macro_step": f1.item(),
                "val_global_step": self.global_step,
            })

        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trả về embedding [B, D] để dùng trong gallery/inference.
        """
        emb, _ = self.model(x, labels=None)
        return emb


class MultiViewPlayerReIDModule(pl.LightningModule):
    """
    Multi-view module: mỗi batch gồm (img_side, img_top, label).
    - ID loss + Triplet loss trên side
    - ID loss + Triplet loss trên top (optional)
    - Consistency loss giữa embedding side & top
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        embedding_dim: int = 512,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_triplet: float = 1.0,
        lambda_consistency: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ReIDNet(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            backbone_name=backbone_name,
            pretrained=True,
            arcface_s=arcface_s,
            arcface_m=arcface_m,
        )

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3)
        
        # HParams
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_triplet = lambda_triplet
        self.lambda_consistency = lambda_consistency

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(x, labels)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        img_side, img_top, labels = batch

        # side
        emb_side, logits_side = self(img_side, labels)
        loss_id_side = self.ce_loss(logits_side, labels)
        loss_tri_side = self.triplet_loss(emb_side, labels)

        # top
        emb_top, logits_top = self(img_top, labels)
        loss_id_top = self.ce_loss(logits_top, labels)
        loss_tri_top = self.triplet_loss(emb_top, labels)

        # consistency: L2 hoặc 1 - cos sim
        consistency = F.mse_loss(emb_side, emb_top)

        loss = (
            loss_id_side
            + loss_tri_side * self.lambda_triplet
            +  0.5 * (loss_id_top + loss_tri_top * self.lambda_triplet)
            + consistency * self.lambda_consistency
        )

        preds_side = torch.argmax(logits_side, dim=1)
        acc_side = (preds_side == labels).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_id_side", loss_id_side, on_epoch=True)
        self.log("train/loss_tri_side", loss_tri_side, on_epoch=True)
        self.log("train/consistency", consistency, on_epoch=True)
        self.log("train/acc_side", acc_side, on_epoch=True, prog_bar=True)

        # Log to wandb every 10 steps
        if batch_idx % 10 == 0:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/loss_id_side_step": loss_id_side.item(),
                "train/loss_tri_side_step": loss_tri_side.item(),
                "train/loss_id_top_step": loss_id_top.item(),
                "train/loss_tri_top_step": loss_tri_top.item(),
                "train/consistency_step": consistency.item(),
                "train/acc_side_step": acc_side.item(),
                "global_step": self.global_step,
            })

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        img_side, img_top, labels = batch

        emb_side, logits_side = self(img_side, labels)
        emb_top, logits_top = self(img_top, labels)

        loss_id_side = self.ce_loss(logits_side, labels)
        loss_tri_side = self.triplet_loss(emb_side, labels)
        loss_id_top = self.ce_loss(logits_top, labels)
        loss_tri_top = self.triplet_loss(emb_top, labels)
        consistency = F.mse_loss(emb_side, emb_top)

        loss = (
            loss_id_side
            + loss_tri_side * self.lambda_triplet
            + loss_id_top
            + loss_tri_top * self.lambda_triplet
            + consistency * self.lambda_consistency
        )

        preds_side = torch.argmax(logits_side, dim=1)
        acc_side = (preds_side == labels).float().mean()
        f1_side = macro_f1(preds_side, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc_side", acc_side, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro_side", f1_side, on_epoch=True, prog_bar=True)
        self.log("val/consistency", consistency, on_epoch=True)

        # Log to wandb every 10 steps
        if batch_idx % 10 == 0:
            wandb.log({
                "val/loss_step": loss.item(),
                "val/acc_side_step": acc_side.item(),
                "val/f1_macro_side_step": f1_side.item(),
                "val/consistency_step": consistency.item(),
                "val_global_step": self.global_step,
            })

        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
