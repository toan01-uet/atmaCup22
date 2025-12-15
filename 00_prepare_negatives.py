"""
Offline script to pre-generate negative samples (background + partial).
Run this once before training to avoid regenerating negatives every time.

Usage:
    python 00_prepare_negatives.py
    
This will create train_negatives.csv which can be loaded quickly during training.
"""

import pandas as pd
import argparse
import logging
from pathlib import Path

from src.utils.utils import (
    load_train_meta,
    build_tracks,
    clean_tracks,
    generate_background_negatives,
    generate_partial_negatives,
    BBoxSample,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_negatives(
    train_meta_path: str,
    img_root: str,
    output_csv: str = "train_negatives.csv",
    num_bg_per_frame: int = 1,
    num_partial_per_box: int = 1,
    max_iou_partial: float = 0.4,
    bbox_mode: str = "drop",
):
    """
    Generate background and partial negative samples and save to CSV.
    
    Args:
        train_meta_path: Path to train_meta.csv
        img_root: Root directory for images
        output_csv: Output CSV file path
        num_bg_per_frame: Number of background negatives per frame
        num_partial_per_box: Number of partial negatives per bounding box
        max_iou_partial: Maximum IoU threshold for partial negatives
        bbox_mode: Mode for handling bad bboxes ('drop' or 'interp')
    """
    logger.info(f"Loading train metadata from {train_meta_path}...")
    samples = load_train_meta(train_meta_path, img_root)
    logger.info(f"Loaded {len(samples)} samples")
    
    logger.info(f"Building tracks with bbox_mode={bbox_mode}...")
    tracks = build_tracks(samples)
    logger.info(f"Built {len(tracks)} tracks")
    
    cleaned_samples = clean_tracks(tracks, mode=bbox_mode)
    logger.info(f"Cleaned to {len(cleaned_samples)} samples")
    
    # Generate background negatives
    logger.info(f"Generating background negatives (num_bg_per_frame={num_bg_per_frame})...")
    bg_neg = generate_background_negatives(
        samples=cleaned_samples,
        num_bg_per_frame=num_bg_per_frame,
    )
    logger.info(f"Generated {len(bg_neg)} background negatives")
    
    # Generate partial negatives
    logger.info(f"Generating partial negatives (num_partial_per_box={num_partial_per_box})...")
    partial_neg = generate_partial_negatives(
        samples=cleaned_samples,
        num_partial_per_box=num_partial_per_box,
        max_iou_with_gt=max_iou_partial,
    )
    logger.info(f"Generated {len(partial_neg)} partial negatives")
    
    # Combine all negatives
    neg_samples = bg_neg + partial_neg
    logger.info(f"Total negative samples: {len(neg_samples)}")
    
    # Convert to DataFrame and save
    logger.info(f"Converting to DataFrame and saving to {output_csv}...")
    rows = []
    for s in neg_samples:
        rows.append({
            'quarter': s.quarter,
            'angle': s.angle,
            'session': s.session,
            'frame': s.frame,
            'x': s.x,
            'y': s.y,
            'w': s.w,
            'h': s.h,
            'label_id': s.label_id,  # Should be -1 for negatives
            'img_path': s.img_path,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    logger.info(f"✓ Successfully saved {len(df)} negative samples to {output_csv}")
    logger.info(f"  - Background negatives: {len(bg_neg)}")
    logger.info(f"  - Partial negatives: {len(partial_neg)}")
    
    # Print some statistics
    logger.info("\nNegative samples statistics:")
    logger.info(f"  - Unique quarters: {df['quarter'].nunique()}")
    logger.info(f"  - Unique angles: {df['angle'].nunique()}")
    logger.info(f"  - Unique sessions: {df['session'].nunique()}")
    logger.info(f"  - Total frames: {df['frame'].nunique()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate negative samples for training"
    )
    parser.add_argument(
        "--train_meta",
        type=str,
        default="inputs/atmaCup22_2nd_meta/train_meta.csv",
        help="Path to train_meta.csv"
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="inputs/images",
        help="Root directory for images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inputs/train_negatives.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--num_bg_per_frame",
        type=int,
        default=1,
        help="Number of background negatives per frame"
    )
    parser.add_argument(
        "--num_partial_per_box",
        type=int,
        default=1,
        help="Number of partial negatives per bounding box"
    )
    parser.add_argument(
        "--max_iou_partial",
        type=float,
        default=0.4,
        help="Maximum IoU threshold for partial negatives"
    )
    parser.add_argument(
        "--bbox_mode",
        type=str,
        default="drop",
        choices=["drop", "interp"],
        help="Mode for handling bad bboxes"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate negatives
    prepare_negatives(
        train_meta_path=args.train_meta,
        img_root=args.img_root,
        output_csv=args.output,
        num_bg_per_frame=args.num_bg_per_frame,
        num_partial_per_box=args.num_partial_per_box,
        max_iou_partial=args.max_iou_partial,
        bbox_mode=args.bbox_mode,
    )


if __name__ == "__main__":
    main()
