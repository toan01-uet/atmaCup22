import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datamodule import PlayerReIDDataModule
from pl_module import PlayerReIDModule

if __name__ == "__main__":
    dm = PlayerReIDDataModule(
        train_meta_path="train_meta.csv",
        img_root="train_images",
        batch_size=64,
        num_workers=8,
        train_ratio=0.8,   # chỉ để tách val monitor F1
        bbox_mode="drop",
        image_size=224,
    )
    dm.setup()
    num_classes = dm.num_classes

    model = PlayerReIDModule(
        num_classes=num_classes,
        backbone_name="resnet18",
        embedding_dim=512,
        arcface_s=30.0,
        arcface_m=0.5,
        lr=1e-4,
        weight_decay=1e-5,
        lambda_triplet=1.0,
    )

    ckpt_cb = ModelCheckpoint(
        monitor="val/f1_macro",
        mode="max",
        save_top_k=1,
        filename="reid-{epoch}-{val_f1_macro:.3f}",
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_cb, lr_cb],
    )

    trainer.fit(model, datamodule=dm)
