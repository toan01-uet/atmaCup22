"""
Example: Using Multi-View Fine-tuning

This example shows how to add Stage 2 multi-view fine-tuning to your training pipeline.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import PlayerReIDDataModuleCV, PlayerReIDDataModuleMV
from src.models.pl_module import PlayerReIDModule
from src.utils.utils import (
    load_train_meta,
    build_tracks,
    clean_tracks,
    make_label_folds,
    build_multiview_pairs
)


def train_with_finetuning(
    train_meta_path: str,
    img_root: str,
    fold_idx: int = 0,
    n_folds: int = 5,
    unknown_per_fold: int = 2,
    cv_seed: int = 42,
):
    """
    Complete training pipeline with optional Stage 2 fine-tuning.
    
    Args:
        train_meta_path: Path to train_meta.csv
        img_root: Root directory for images
        fold_idx: Current fold index (0-4)
        n_folds: Total number of folds
        unknown_per_fold: Number of unknown labels per fold
        cv_seed: Random seed for CV splits
    """
    
    print("="*80)
    print(f"TRAINING FOLD {fold_idx} WITH MULTI-VIEW FINE-TUNING")
    print("="*80)
    
    # ========================================================
    # STAGE 1: Main Training
    # ========================================================
    print("\n### STAGE 1: Main Training ###\n")
    
    # Setup datamodule
    dm = PlayerReIDDataModuleCV(
        train_meta_path=train_meta_path,
        img_root=img_root,
        batch_size=64,
        num_workers=8,
        train_ratio=0.8,
        bbox_mode="drop",
        image_size=224,
        n_folds=n_folds,
        fold_idx=fold_idx,
        unknown_per_fold=unknown_per_fold,
        cv_seed=cv_seed,
    )
    dm.setup()
    
    # Initialize model
    model = PlayerReIDModule(
        num_classes=dm.num_classes,
        backbone_name="resnet18",
        embedding_dim=512,
        arcface_s=30.0,
        arcface_m=0.5,
        lr=1e-4,
        weight_decay=1e-5,
        lambda_triplet=1.0,
    )
    
    # Setup callbacks
    ckpt_cb = ModelCheckpoint(
        monitor="val/f1_macro",
        mode="max",
        save_top_k=1,
        filename=f"reid-fold{fold_idx}-{{epoch}}-{{val_f1_macro:.3f}}",
    )
    
    # Train Stage 1
    trainer = pl.Trainer(
        max_epochs=12,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_cb, LearningRateMonitor()],
        logger=WandbLogger(project="atmacup22-reid", name=f"fold{fold_idx}"),
    )
    
    print("Starting Stage 1 training...")
    trainer.fit(model, datamodule=dm)
    
    print(f"\nStage 1 Complete!")
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"Best F1: {ckpt_cb.best_model_score:.4f}")
    
    # ========================================================
    # STAGE 2: Multi-View Fine-tuning
    # ========================================================
    print("\n### STAGE 2: Multi-View Fine-tuning ###\n")
    
    # Load best model from Stage 1
    model_ft = PlayerReIDModule.load_from_checkpoint(
        ckpt_cb.best_model_path,
        num_classes=dm.num_classes,
        lr=3e-5,  # Lower learning rate
        lambda_consistency=0.1,  # Add consistency loss
    )
    
    # Build multi-view pairs
    print("Building multi-view pairs...")
    all_samples = load_train_meta(train_meta_path, img_root)
    all_tracks = build_tracks(all_samples)
    all_cleaned = clean_tracks(all_tracks, mode="drop")
    
    # Filter for current fold's known labels
    all_labels = sorted({s.label_id for s in all_cleaned})
    folds_unknown = make_label_folds(
        unique_labels=all_labels,
        n_folds=n_folds,
        unknown_per_fold=unknown_per_fold,
        seed=cv_seed,
    )
    unknown_labels = set(folds_unknown[fold_idx])
    known_labels = [lab for lab in all_labels if lab not in unknown_labels]
    fold_samples = [s for s in all_cleaned if s.label_id in known_labels]
    
    # Create pairs
    mv_pairs = build_multiview_pairs(fold_samples)
    print(f"Found {len(mv_pairs)} multi-view pairs")
    
    if len(mv_pairs) == 0:
        print("⚠️ No multi-view pairs found. Skipping Stage 2.")
        return ckpt_cb.best_model_path
    
    # Setup multi-view datamodule
    dm_mv = PlayerReIDDataModuleMV(
        pairs=mv_pairs,
        batch_size=32,
        num_workers=4,
        train_ratio=0.8,
        image_size=224,
    )
    dm_mv.setup()
    
    # Setup fine-tuning callbacks
    ckpt_cb_ft = ModelCheckpoint(
        monitor="val/f1_macro",
        mode="max",
        save_top_k=1,
        filename=f"reid-fold{fold_idx}-finetuned-{{epoch}}-{{val_f1_macro:.3f}}",
    )
    
    # Fine-tune
    trainer_ft = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_cb_ft, LearningRateMonitor()],
        logger=WandbLogger(project="atmacup22-reid", name=f"fold{fold_idx}_ft"),
    )
    
    print("Starting Stage 2 fine-tuning...")
    trainer_ft.fit(model_ft, datamodule=dm_mv)
    
    print(f"\nStage 2 Complete!")
    print(f"Best checkpoint: {ckpt_cb_ft.best_model_path}")
    print(f"Best F1: {ckpt_cb_ft.best_model_score:.4f}")
    
    print("\n" + "="*80)
    print(f"FOLD {fold_idx} COMPLETE")
    print("="*80)
    
    return ckpt_cb_ft.best_model_path


if __name__ == "__main__":
    import sys
    
    # Configuration
    train_meta_path = "inputs/atmaCup22_2nd_meta/train_meta.csv"
    img_root = "inputs/images"
    
    # Train single fold (for testing)
    fold_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    best_checkpoint = train_with_finetuning(
        train_meta_path=train_meta_path,
        img_root=img_root,
        fold_idx=fold_idx,
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model: {best_checkpoint}")
    
    # Usage:
    # python example_multiview_finetuning.py 0  # Train fold 0
    # python example_multiview_finetuning.py 1  # Train fold 1
    # etc.
