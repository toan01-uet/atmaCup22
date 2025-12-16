"""
CV-based inference pipeline using unified fold metadata.
This script uses the SAME fold splits as training for consistency.

Usage:
    python 02_inference_cv.py --fold 0
    python 02_inference_cv.py --fold 0 --checkpoint path/to/model.ckpt
"""

import os
import argparse
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

from src.inference import (
    build_open_set_splits_cv,
    build_val_tracklets_from_unknown,
    tune_threshold,
)
from src.utils.utils import (
    load_test_meta,
    build_test_tracklets,
    build_gallery,
    get_tracklet_embedding,
    compute_cosine_similarity,
)
from src.models.pl_module import PlayerReIDModule
from src.utils.logger import get_logger


def run_inference_cv(
    fold_idx: int,
    ckpt_path: str,
    fold_meta_dir: str = "outputs/fold_metadata",
    neg_csv_path: str = "inputs/train_negatives.csv",
    test_meta_path: str = "inputs/atmaCup22_2nd/test_meta.csv",
    test_img_root: str = "inputs/atmaCup22_2nd/crops",
    submission_path: str = "outputs/submission_fold0.csv",
    device: str = "cuda",
    image_size: int = 224,
    iou_threshold: float = 0.3,
    max_frame_gap: int = 30,
    logger=None,
):
    """
    Run inference for a single CV fold using saved fold metadata.
    
    Args:
        fold_idx: Index of the fold (0-4)
        ckpt_path: Path to model checkpoint
        fold_meta_dir: Directory containing saved fold metadata
        neg_csv_path: Path to pre-computed negatives CSV
        test_meta_path: Path to test_meta.csv
        test_img_root: Root directory for test images
        submission_path: Output submission CSV path
        device: Device to run inference on
        image_size: Image size for inference
        iou_threshold: IoU threshold for tracklet matching
        max_frame_gap: Maximum frame gap for tracklet matching
        logger: Logger instance
    """
    if logger:
        logger.info("="*80)
        logger.info(f"CV INFERENCE - FOLD {fold_idx}")
        logger.info("="*80)
    
    # 1. Load model
    if logger:
        logger.info(f"\nLoading model from checkpoint...")
        logger.info(f"Checkpoint: {ckpt_path}")
    device = torch.device(device)
    pl_module = PlayerReIDModule.load_from_checkpoint(ckpt_path)
    pl_module.eval().to(device)
    if logger:
        logger.info(f"✓ Model loaded on device: {device}")

    # 2. Load fold metadata (train + val samples, unknown labels)
    if logger:
        logger.info(f"\nLoading fold {fold_idx} metadata from {fold_meta_dir}...")
    
    known_samples, unknown_samples, known_labels, unknown_labels = build_open_set_splits_cv(
        fold_idx=fold_idx,
        fold_meta_dir=fold_meta_dir,
        neg_csv_path=neg_csv_path,
        add_negatives=True,
    )
    
    if logger:
        logger.info(f"✓ Known samples: {len(known_samples)} (from train+val)")
        logger.info(f"✓ Known labels: {len(known_labels)}")
        logger.info(f"✓ Unknown samples: {len(unknown_samples)} (players + negatives)")
        logger.info(f"✓ Unknown labels: {len(unknown_labels)}")

    # 3. Build gallery from known samples (side angle only)
    known_samples_side = [s for s in known_samples if s.angle == "side"]
    if logger:
        logger.info(f"\nBuilding gallery from {len(known_samples_side)} side-angle samples...")
    
    gallery = build_gallery(
        pl_module,
        samples=known_samples_side,
        batch_size=64,
        image_size=image_size,
        device=device,
    )
    
    if logger:
        logger.info(f"✓ Gallery built with {len(gallery)} identities")

    # 4. Build unknown tracklets and tune threshold
    if logger:
        logger.info(f"\nBuilding unknown tracklets for threshold tuning...")
    
    unknown_tracklets = build_val_tracklets_from_unknown(unknown_samples)
    
    if logger:
        logger.info(f"✓ Built {len(unknown_tracklets)} unknown tracklets")
        logger.info("Tuning threshold...")
    
    threshold = tune_threshold(
        pl_module,
        gallery,
        known_samples=known_samples,
        unknown_tracklets=unknown_tracklets,
        image_size=image_size,
        device=device,
    )
    
    if logger:
        logger.info(f"✓ Best threshold: {threshold:.4f}")

    # 5. Load test data and build tracklets
    if logger:
        logger.info(f"\nLoading test metadata from {test_meta_path}...")
    
    test_bboxes = load_test_meta(test_meta_path, test_img_root)
    
    if logger:
        logger.info(f"✓ Loaded {len(test_bboxes)} test bounding boxes")
        logger.info(f"\nBuilding test tracklets (IoU={iou_threshold}, max_gap={max_frame_gap})...")
    
    test_tracklets = build_test_tracklets(
        test_bboxes,
        iou_threshold=iou_threshold,
        max_frame_gap=max_frame_gap,
    )
    
    if logger:
        logger.info(f"✓ Built {len(test_tracklets)} test tracklets")

    # 6. Get embeddings and predict labels for each tracklet
    if logger:
        logger.info("\nRunning inference on test tracklets...")
    
    for tr in tqdm(test_tracklets, desc="Inference"):
        # Get tracklet embedding
        emb = get_tracklet_embedding(
            pl_module,
            tr.bboxes,
            batch_size=64,
            image_size=image_size,
            device=device,
        )
        
        # Find best match in gallery
        best_label = -1
        best_sim = -1.0
        
        for label_id, gallery_emb in gallery.items():
            sim = compute_cosine_similarity(emb, gallery_emb)
            if sim > best_sim:
                best_sim = sim
                best_label = label_id
        
        # Apply threshold
        if best_sim < threshold:
            tr.pred_label = -1
        else:
            tr.pred_label = best_label

    # 7. Apply frame constraint: multiple overlapping bboxes can have same ID
    if logger:
        logger.info("\nApplying frame constraint (allowing multiple bboxes with same ID)...")
    
    # Group bboxes by frame
    from collections import defaultdict
    frame_bboxes = defaultdict(list)
    
    for tr in test_tracklets:
        for bbox in tr.bboxes:
            frame_bboxes[(bbox.quarter, bbox.angle, bbox.session, bbox.frame)].append(
                (bbox, tr.pred_label)
            )
    
    # For each frame, assign labels (can be duplicate if bboxes overlap)
    final_predictions = []
    
    for frame_key, bbox_label_pairs in frame_bboxes.items():
        # Sort by bbox index to maintain order
        bbox_label_pairs.sort(key=lambda x: x[0].idx)
        
        for bbox, pred_label in bbox_label_pairs:
            final_predictions.append({
                'idx': bbox.idx,
                'label_id': pred_label
            })
    
    # 8. Create submission DataFrame
    submission_df = pd.DataFrame(final_predictions)
    submission_df = submission_df.sort_values('idx').reset_index(drop=True)
    
    # Ensure all test indices are present
    all_indices = set(range(len(test_bboxes)))
    pred_indices = set(submission_df['idx'].values)
    missing_indices = all_indices - pred_indices
    
    if missing_indices:
        if logger:
            logger.warning(f"⚠ Found {len(missing_indices)} missing indices, filling with -1")
        missing_df = pd.DataFrame([
            {'idx': idx, 'label_id': -1} for idx in sorted(missing_indices)
        ])
        submission_df = pd.concat([submission_df, missing_df], ignore_index=True)
        submission_df = submission_df.sort_values('idx').reset_index(drop=True)
    
    # 9. Save submission
    submission_df.to_csv(submission_path, index=False)
    
    if logger:
        logger.info(f"\n✓ Submission saved to {submission_path}")
        logger.info(f"  Total predictions: {len(submission_df)}")
        logger.info(f"  Known assignments: {(submission_df['label_id'] != -1).sum()}")
        logger.info(f"  Unknown assignments: {(submission_df['label_id'] == -1).sum()}")
        logger.info(f"  Unique predicted labels: {submission_df[submission_df['label_id'] != -1]['label_id'].nunique()}")
    
    return submission_df


def main():
    parser = argparse.ArgumentParser(
        description="Run CV-based inference using fold metadata"
    )
    
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold index (0-4)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (if not provided, will look in default location)"
    )
    parser.add_argument(
        "--fold_meta_dir",
        type=str,
        default="outputs/fold_metadata",
        help="Directory containing fold metadata"
    )
    parser.add_argument(
        "--neg_csv",
        type=str,
        default="inputs/train_negatives.csv",
        help="Path to pre-computed negatives CSV"
    )
    parser.add_argument(
        "--test_meta",
        type=str,
        default="inputs/atmaCup22_2nd/test_meta.csv",
        help="Path to test_meta.csv"
    )
    parser.add_argument(
        "--test_img_root",
        type=str,
        default="inputs/atmaCup22_2nd/crops",
        help="Root directory for test images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output submission CSV path (default: outputs/submission_fold{fold}.csv)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for tracklet matching"
    )
    parser.add_argument(
        "--max_frame_gap",
        type=int,
        default=30,
        help="Maximum frame gap for tracklet matching"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find checkpoint if not provided
    if args.checkpoint is None:
        # Look for checkpoint in default wandb directory
        ckpt_dir = Path(script_dir) / "atmacup22-reid"
        if ckpt_dir.exists():
            # Find the fold checkpoint
            ckpts = list(ckpt_dir.glob(f"*/checkpoints/*fold{args.fold}*.ckpt"))
            if ckpts:
                args.checkpoint = str(ckpts[0])
                print(f"Found checkpoint: {args.checkpoint}")
            else:
                print(f"Error: No checkpoint found for fold {args.fold}")
                return
        else:
            print(f"Error: Please provide --checkpoint path")
            return
    
    # Set output path
    if args.output is None:
        args.output = os.path.join(script_dir, "outputs", f"submission_fold{args.fold}.csv")
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    outputs_dir = Path(script_dir) / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    logger = get_logger(f"inference_fold{args.fold}", file_dir=outputs_dir)
    
    logger.info("="*80)
    logger.info("CV-BASED INFERENCE PIPELINE")
    logger.info("="*80)
    logger.info(f"Fold: {args.fold}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Fold metadata dir: {args.fold_meta_dir}")
    logger.info(f"Test metadata: {args.test_meta}")
    logger.info(f"Output: {args.output}")
    
    # Run inference
    run_inference_cv(
        fold_idx=args.fold,
        ckpt_path=args.checkpoint,
        fold_meta_dir=args.fold_meta_dir,
        neg_csv_path=args.neg_csv,
        test_meta_path=args.test_meta,
        test_img_root=args.test_img_root,
        submission_path=args.output,
        device=args.device,
        iou_threshold=args.iou_threshold,
        max_frame_gap=args.max_frame_gap,
        logger=logger,
    )
    
    logger.info("\n" + "="*80)
    logger.info("INFERENCE COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
