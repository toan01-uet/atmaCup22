"""
Cross-validation training with optional Stage 2 Multi-View Fine-tuning.

This script trains models for all CV folds with an optional fine-tuning stage
that adds multi-view consistency loss between side and top camera angles.

Usage:
    # Train all folds with fine-tuning
    python 01_run_cv_train_with_finetune.py
    
    # Skip fine-tuning (set ENABLE_FINETUNE=False in code)
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from src.data.datamodule import PlayerReIDDataModuleCV, PlayerReIDDataModuleMV
from src.models.pl_module import PlayerReIDModule
from src.utils.logger import get_logger
from src.utils.utils import (
    load_train_meta,
    build_tracks,
    clean_tracks,
    make_label_folds,
    build_multiview_pairs
)

# Configuration
ENABLE_FINETUNE = True  # Set to False to skip Stage 2
FINETUNE_EPOCHS = 5
FINETUNE_LR = 3e-5
FINETUNE_CONSISTENCY_WEIGHT = 0.1

if __name__ == "__main__":
    # Convert to absolute paths to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup logging
    outputs_dir = Path(script_dir) / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    logger = get_logger("train_finetune", file_dir=outputs_dir)
    
    logger.info("="*80)
    logger.info("Starting Training Pipeline with Optional Fine-tuning")
    logger.info("="*80)
    
    train_meta_path = os.path.join(script_dir, "inputs", "atmaCup22_2nd_meta", "train_meta.csv")
    img_root = os.path.join(script_dir, "inputs", "images")
    
    logger.info(f"Train metadata path: {train_meta_path}")
    logger.info(f"Image root: {img_root}")
    
    # Cross-validation configuration
    N_FOLDS = 5
    UNKNOWN_PER_FOLD = 2
    CV_SEED = 42
    
    logger.info(f"\nCross-Validation Configuration:")
    logger.info(f"  - Number of folds: {N_FOLDS}")
    logger.info(f"  - Unknown labels per fold: {UNKNOWN_PER_FOLD}")
    logger.info(f"  - CV seed: {CV_SEED}")
    logger.info(f"  - Batch size: 64")
    logger.info(f"  - Num workers: 8")
    logger.info(f"  - Image size: 224")
    logger.info(f"  - BBox mode: drop")
    logger.info(f"  - Stage 2 Fine-tuning: {'ENABLED' if ENABLE_FINETUNE else 'DISABLED'}")
    
    # Loop through all folds
    for FOLD_IDX in range(N_FOLDS):
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING FOLD {FOLD_IDX}/{N_FOLDS-1}")
        logger.info("="*80)
        
        # ============================================================
        # Stage 1: Main Training (Side-view focused)
        # ============================================================
        logger.info(f"\n### Stage 1: Main Training - Fold {FOLD_IDX} ###")
        logger.info(f"\nInitializing DataModule for Fold {FOLD_IDX}...")

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
            max_epochs=12,
            accelerator="gpu",
            devices=1,
            callbacks=[ckpt_cb, lr_cb],
            logger=wandb_logger,
            log_every_n_steps=10,
        )
        logger.info(f"Max epochs: 12")
        logger.info(f"Accelerator: gpu, devices: 1")
        logger.info(f"Log every n steps: 10")

        logger.info("\n" + "="*80)
        logger.info(f"Starting Stage 1 Training - Fold {FOLD_IDX}")
        logger.info("="*80)
        trainer.fit(model, datamodule=dm)
        
        logger.info("\n" + "="*80)
        logger.info(f"Fold {FOLD_IDX} Stage 1 Training Complete!")
        logger.info(f"Best checkpoint: {ckpt_cb.best_model_path}")
        logger.info(f"Best val/f1_macro: {ckpt_cb.best_model_score:.4f}")
        logger.info("="*80)
        
        # ============================================================
        # Stage 2: Multi-View Fine-tuning (Optional)
        # ============================================================
        if not ENABLE_FINETUNE:
            logger.info("\n⏭️  Skipping Stage 2 fine-tuning (ENABLE_FINETUNE=False)")
            continue
            
        logger.info("\n" + "="*80)
        logger.info(f"### Stage 2: Multi-View Fine-tuning - Fold {FOLD_IDX} ###")
        logger.info("="*80)
        
        # Load best checkpoint from Stage 1
        logger.info(f"\nLoading best model from Stage 1...")
        model_ft = PlayerReIDModule.load_from_checkpoint(
            ckpt_cb.best_model_path,
            num_classes=num_classes,
            backbone_name="resnet18",
            embedding_dim=512,
            arcface_s=30.0,
            arcface_m=0.5,
            lr=FINETUNE_LR,
            weight_decay=1e-5,
            lambda_triplet=1.0,
            lambda_consistency=FINETUNE_CONSISTENCY_WEIGHT,
        )
        logger.info(f"Model loaded for fine-tuning")
        logger.info(f"Fine-tuning LR: {FINETUNE_LR}")
        logger.info(f"Consistency loss weight: {FINETUNE_CONSISTENCY_WEIGHT}")
        
        # Build multi-view pairs
        logger.info(f"\nBuilding multi-view pairs from training samples...")
        
        # Reload data and filter for current fold
        all_samples = load_train_meta(train_meta_path, img_root)
        all_tracks = build_tracks(all_samples)
        all_cleaned = clean_tracks(all_tracks, mode="drop")
        
        # Get labels for this fold
        all_labels = sorted({s.label_id for s in all_cleaned})
        folds_unknown = make_label_folds(
            unique_labels=all_labels,
            n_folds=N_FOLDS,
            unknown_per_fold=UNKNOWN_PER_FOLD,
            seed=CV_SEED,
        )
        unknown_labels = set(folds_unknown[FOLD_IDX])
        known_labels = [lab for lab in all_labels if lab not in unknown_labels]
        
        # Filter samples for known labels only
        fold_samples = [s for s in all_cleaned if s.label_id in known_labels]
        
        # Build multi-view pairs
        mv_pairs = build_multiview_pairs(fold_samples)
        logger.info(f"Built {len(mv_pairs)} multi-view pairs (side+top)")
        
        if len(mv_pairs) == 0:
            logger.warning("⚠️ No multi-view pairs found. Skipping Stage 2 fine-tuning.")
            continue
        
        # Create multi-view datamodule
        dm_mv = PlayerReIDDataModuleMV(
            pairs=mv_pairs,
            batch_size=32,  # Smaller batch size for fine-tuning
            num_workers=4,
            train_ratio=0.8,
            image_size=224,
            logger=logger,
        )
        dm_mv.setup()
        
        logger.info(f"Train pairs: {len(dm_mv.train_pairs)}")
        logger.info(f"Val pairs: {len(dm_mv.val_pairs)}")
        
        # Setup fine-tuning checkpoint callback
        ckpt_cb_ft = ModelCheckpoint(
            monitor="val/f1_macro",
            mode="max",
            save_top_k=1,
            filename=f"reid-fold{FOLD_IDX}-finetuned-{{epoch}}-{{val_f1_macro:.3f}}",
        )
        
        # Setup fine-tuning wandb logger
        wandb_logger_ft = WandbLogger(
            project="atmacup22-reid",
            name=f"resnet18_cv_fold{FOLD_IDX}_finetune",
            log_model=False,
        )
        
        # Fine-tuning trainer with fewer epochs
        trainer_ft = pl.Trainer(
            max_epochs=FINETUNE_EPOCHS,
            accelerator="gpu",
            devices=1,
            callbacks=[ckpt_cb_ft, lr_cb],
            logger=wandb_logger_ft,
            log_every_n_steps=5,
        )
        
        logger.info("\nStarting Stage 2 fine-tuning...")
        logger.info(f"Fine-tuning epochs: {FINETUNE_EPOCHS}")
        logger.info(f"Learning rate: {FINETUNE_LR} (reduced)")
        logger.info(f"Consistency loss weight: {FINETUNE_CONSISTENCY_WEIGHT}")
        
        trainer_ft.fit(model_ft, datamodule=dm_mv)
        
        logger.info("\n" + "="*80)
        logger.info(f"Fold {FOLD_IDX} Stage 2 Fine-tuning Complete!")
        logger.info(f"Best checkpoint: {ckpt_cb_ft.best_model_path}")
        logger.info(f"Best val/f1_macro: {ckpt_cb_ft.best_model_score:.4f}")
        logger.info("="*80)
    
    logger.info("\n" + "="*80)
    logger.info("ALL FOLDS TRAINING COMPLETE!")
    logger.info("="*80)
