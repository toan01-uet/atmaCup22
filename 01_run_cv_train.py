import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from src.data.datamodule import PlayerReIDDataModuleCV
from src.models.pl_module import PlayerReIDModule
from src.utils.logger import get_logger

if __name__ == "__main__":
    # Convert to absolute paths to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup logging
    outputs_dir = Path(script_dir) / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    logger = get_logger("train", file_dir=outputs_dir)
    
    logger.info("="*80)
    logger.info("Starting Training Pipeline")
    logger.info("="*80)
    
    train_meta_path = os.path.join(script_dir, "inputs", "atmaCup22_2nd_meta", "train_meta.csv")
    img_root = os.path.join(script_dir, "inputs", "images")
    
    logger.info(f"Train metadata path: {train_meta_path}")
    logger.info(f"Image root: {img_root}")
    
    logger.info("\nInitializing Cross-Validation DataModule...")
    FOLD_IDX = 0  # 0..4
    N_FOLDS = 5
    UNKNOWN_PER_FOLD = 2
    CV_SEED = 42
    
    logger.info(f"Cross-Validation Configuration:")
    logger.info(f"  - Number of folds: {N_FOLDS}")
    logger.info(f"  - Current fold: {FOLD_IDX}")
    logger.info(f"  - Unknown labels per fold: {UNKNOWN_PER_FOLD}")
    logger.info(f"  - CV seed: {CV_SEED}")
    logger.info(f"  - Batch size: 64")
    logger.info(f"  - Num workers: 8")
    logger.info(f"  - Image size: 224")
    logger.info(f"  - BBox mode: drop")

    dm = PlayerReIDDataModuleCV(
        train_meta_path=train_meta_path,
        img_root=img_root,
        batch_size=64,
        num_workers=8,
        train_ratio=0.8,
        bbox_mode="drop",
        image_size=224,
        n_folds=N_FOLDS,
        fold_idx=FOLD_IDX,
        unknown_per_fold=UNKNOWN_PER_FOLD,
        cv_seed=CV_SEED,
        logger=logger,
    )
    logger.info("Setting up data splits...")
    dm.setup()
    num_classes = dm.num_classes
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train samples: {len(dm.train_samples)}")
    logger.info(f"Val samples: {len(dm.val_samples)}")

    logger.info("\nInitializing Model...")
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
    logger.info(f"Model backbone: resnet18")
    logger.info(f"Embedding dimension: 512")
    logger.info(f"Learning rate: 1e-4")
    logger.info(f"Lambda triplet: 1.0")

    logger.info("\nSetting up callbacks...")
    ckpt_cb = ModelCheckpoint(
        monitor="val/f1_macro",
        mode="max",
        save_top_k=1,
        filename=f"reid-fold{FOLD_IDX}-{{epoch}}-{{val_f1_macro:.3f}}",
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    logger.info("Checkpoint callback: monitoring val/f1_macro (max)")

    logger.info("\nInitializing WandB Logger...")
    wandb_logger = WandbLogger(
        project="atmacup22-reid",
        name=f"resnet18_cv_fold{FOLD_IDX}",
        log_model=False,
    )
    logger.info(f"WandB project: atmacup22-reid, run: resnet18_cv_fold{FOLD_IDX}")

    logger.info("\nInitializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_cb, lr_cb],
        logger=wandb_logger,
        log_every_n_steps=10,
    )
    logger.info(f"Max epochs: 20")
    logger.info(f"Accelerator: gpu, devices: 1")
    logger.info(f"Log every n steps: 10")

    logger.info("\n" + "="*80)
    logger.info(f"Starting Cross-Validation Training - Fold {FOLD_IDX}")
    logger.info("="*80)
    trainer.fit(model, datamodule=dm)
    
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info(f"Best checkpoint: {ckpt_cb.best_model_path}")
    logger.info(f"Best val/f1_macro: {ckpt_cb.best_model_score:.4f}")
    logger.info("="*80)
