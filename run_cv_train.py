import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import torch

from src.data.datamodule import PlayerReIDDataModuleCV
from src.models.pl_module import PlayerReIDModule
from src.utils.logger import get_logger

"""
Cross-validation training pipeline for player re-identification.

Data flow:
    1. Run 'python prepare_data.py' to generate:
       - inputs/train_crops_cache/  (cached training crops)
       - inputs/train_negatives.csv (negative samples)
    
    2. This script loads:
       - inputs/atmaCup22_2nd/train_meta.csv  (metadata)
       - inputs/train_crops_cache/            (cached crops)
       - inputs/train_negatives.csv           (negatives)
    
    3. Saves trained models to:
       - outputs/fold_metadata/     (fold splits and samples)
       - wandb/                     (weights and biases logs)

Usage:
    python run_cv_train.py          # Train all 5 folds
"""

def run():
    # Set PyTorch memory management for better GPU utilization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert to absolute paths to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup logging
    outputs_dir = Path(script_dir) / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    logger = get_logger("train", file_dir=outputs_dir)
    
    logger.info("="*80)
    logger.info("Starting Training Pipeline")
    logger.info("="*80)
    
    train_meta_path = os.path.join(script_dir, "inputs", "atmaCup22_2nd", "train_meta.csv")
    img_root = os.path.join(script_dir, "inputs", "train_crops_cache")
    
    logger.info(f"Train metadata path: {train_meta_path}")
    logger.info(f"Image root (cached training crops): {img_root}")
    logger.info(f"Note: Make sure to run 'python prepare_data.py' first to generate {img_root}")
    
    # Cross-validation configuration
    N_FOLDS = 5
    UNKNOWN_PER_FOLD = 2
    CV_SEED = 42
    
    # Memory-optimized configuration
    BATCH_SIZE = 32  # Reduced from 64 to reduce memory usage
    NUM_WORKERS = 4  # Reduced from 8 to reduce memory overhead
    ACCUMULATE_GRAD_BATCHES = 2  # Effective batch size = 32 * 2 = 64
    
    logger.info(f"\nCross-Validation Configuration:")
    logger.info(f"  - Number of folds: {N_FOLDS}")
    logger.info(f"  - Unknown labels per fold: {UNKNOWN_PER_FOLD}")
    logger.info(f"  - CV seed: {CV_SEED}")
    logger.info(f"  - Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUMULATE_GRAD_BATCHES} with accumulation)")
    logger.info(f"  - Num workers: {NUM_WORKERS}")
    logger.info(f"  - Image size: 224")
    logger.info(f"  - BBox mode: drop")
    logger.info(f"  - Mixed precision: enabled (16-mixed)")
    
    # Loop through all folds
    for FOLD_IDX in range(N_FOLDS):
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING FOLD {FOLD_IDX}/{N_FOLDS-1}")
        logger.info("="*80)
        
        logger.info(f"\nInitializing DataModule for Fold {FOLD_IDX}...")

        dm = PlayerReIDDataModuleCV(
        train_meta_path=train_meta_path,
        img_root=img_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
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
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        precision="16-mixed",  # Enable mixed precision training
        )
        logger.info(f"Max epochs: 12")
        logger.info(f"Accelerator: gpu, devices: 1")
        logger.info(f"Gradient accumulation batches: {ACCUMULATE_GRAD_BATCHES}")
        logger.info(f"Mixed precision: 16-mixed")
        logger.info(f"Log every n steps: 10")

        logger.info("\n" + "="*80)
        logger.info(f"Starting Training - Fold {FOLD_IDX}")
        logger.info("="*80)
        trainer.fit(model, datamodule=dm)
        
        logger.info("\n" + "="*80)
        logger.info(f"Fold {FOLD_IDX} Stage 1 Training Complete!")
        logger.info(f"Best checkpoint: {ckpt_cb.best_model_path}")
        logger.info(f"Best val/f1_macro: {ckpt_cb.best_model_score:.4f}")
        logger.info("="*80)
        
        # Clear GPU memory before next fold
        import torch
        del model, trainer, dm
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory")
        
        # ============================================================
        # Stage 2: Multi-View Fine-tuning (Optional)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info(f"Starting Stage 2: Multi-View Fine-tuning - Fold {FOLD_IDX}")
        logger.info("="*80)
        
        from src.data.datamodule import PlayerReIDDataModuleMV
        from src.utils.utils import build_multiview_pairs
        
        # Load best checkpoint from Stage 1
        logger.info(f"\nLoading best model from Stage 1...")
        model_ft = PlayerReIDModule.load_from_checkpoint(
            ckpt_cb.best_model_path,
            num_classes=num_classes,
            backbone_name="resnet18",
            embedding_dim=512,
            arcface_s=30.0,
            arcface_m=0.5,
            lr=3e-5,  # Much lower learning rate for fine-tuning
            weight_decay=1e-5,
            lambda_triplet=1.0,
            lambda_consistency=0.1,  # Add consistency loss
        )
        logger.info(f"Model loaded for fine-tuning")
        
        # Build multi-view pairs
        logger.info(f"\nBuilding multi-view pairs from training samples...")
        from src.utils.utils import load_train_meta, build_tracks, clean_tracks, make_label_folds
        
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
        else:
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
                max_epochs=5,  # Only 3-5 epochs for fine-tuning
                accelerator="gpu",
                devices=1,
                callbacks=[ckpt_cb_ft, lr_cb],
                logger=wandb_logger_ft,
                log_every_n_steps=5,
            )
            
            logger.info("\nStarting Stage 2 fine-tuning...")
            logger.info(f"Fine-tuning epochs: 5")
            logger.info(f"Learning rate: 3e-5 (reduced)")
            logger.info(f"Consistency loss weight: 0.1")
            
            trainer_ft.fit(model_ft, datamodule=dm_mv)
            
            logger.info("\n" + "="*80)
            logger.info(f"Fold {FOLD_IDX} Stage 2 Fine-tuning Complete!")
            logger.info(f"Best checkpoint: {ckpt_cb_ft.best_model_path}")
            logger.info(f"Best val/f1_macro: {ckpt_cb_ft.best_model_score:.4f}")
            logger.info("="*80)
            
            # Clear GPU memory before next fold
            del model_ft, trainer_ft, dm_mv
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory after fine-tuning")
    
    logger.info("\n" + "="*80)
    logger.info("ALL FOLDS TRAINING COMPLETE!")
    logger.info("="*80)
    
if __name__ == "__main__":
    run()
