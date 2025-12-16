"""
Script to prepare all data before training:
1. Pre-crop and cache all training images from atmaCup22_2nd/images/
2. Generate negative samples (background + partial)

Run this once before training to significantly speed up the training pipeline.

Usage:
    python prepare_data.py                      # Prepare both cache and negatives
    python prepare_data.py --skip-cache         # Only generate negatives
    python prepare_data.py --skip-negatives     # Only prepare cache
    python prepare_data.py --force-recrop       # Force re-crop all images

Data flow:
    inputs/atmaCup22_2nd/images/ (full training frames)
    ↓
    prepare_cropped_cache (crop using bbox from train_meta.csv)
    ↓
    inputs/train_crops_cache/ (cached training crops)
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.utils.utils import (
    load_train_meta, 
    build_tracks, 
    clean_tracks,
    generate_background_negatives,
    generate_partial_negatives,
)
from src.utils.crop_cache import prepare_cropped_cache, get_cache_stats
import pandas as pd


def prepare_crop_cache(
    train_meta_path: str,
    img_root: str,
    cache_dir: str,
    expand_ratio: float = 1.2,
    quality: int = 95,
    force_recrop: bool = False,
):
    """Pre-crop and cache all training images."""
    print("="*80)
    print("STEP 1: Pre-cropping Training Images")
    print("="*80)
    
    print(f"\nTrain metadata: {train_meta_path}")
    print(f"Image root (full training images): {img_root}")
    print(f"Cache directory (output): {cache_dir}")
    
    # Check current cache status
    print("\nChecking cache status...")
    cache_stats = get_cache_stats(cache_dir)
    print(f"Cache exists: {cache_stats['exists']}")
    print(f"Cached files: {cache_stats['num_files']}")
    print(f"Total size: {cache_stats['total_size_mb']:.2f} MB")
    
    if cache_stats['num_files'] > 0 and not force_recrop:
        print("\n⚠ Cache already exists! Use --force-recrop to regenerate.")
        return
    
    # Load and process samples
    print("\nLoading training metadata...")
    samples = load_train_meta(train_meta_path, img_root)
    print(f"Loaded {len(samples)} samples")
    
    # Filter to side angle only (like in training)
    samples = [s for s in samples if s.angle == "side"]
    print(f"Filtered to {len(samples)} side-angle samples")
    
    # Build and clean tracks
    print("\nBuilding and cleaning tracks...")
    tracks = build_tracks(samples)
    print(f"Built {len(tracks)} tracks")
    
    cleaned_samples = clean_tracks(tracks, mode="drop")
    print(f"Cleaned to {len(cleaned_samples)} samples")
    
    # Pre-crop
    print("\n" + "="*80)
    print("Starting cropping process...")
    print("="*80)
    
    cached_samples = prepare_cropped_cache(
        cleaned_samples,
        cache_dir,
        expand_ratio=expand_ratio,
        quality=quality,
        force_recrop=force_recrop
    )
    
    # Final cache stats
    print("\n" + "="*80)
    print("Cropping Complete!")
    print("="*80)
    final_stats = get_cache_stats(cache_dir)
    print(f"✓ Cached files: {final_stats['num_files']}")
    print(f"✓ Total size: {final_stats['total_size_mb']:.2f} MB")
    if final_stats['num_files'] > 0:
        print(f"✓ Average size per crop: {final_stats['total_size_mb']/final_stats['num_files']:.3f} MB")


def prepare_negatives(
    train_meta_path: str,
    img_root: str,
    output_csv: str,
    num_bg_per_frame: int = 1,
    num_partial_per_box: int = 1,
    max_iou_partial: float = 0.4,
    bbox_mode: str = "drop",
    force_regenerate: bool = False,
):
    """Generate background and partial negative samples and save to CSV."""
    print("\n" + "="*80)
    print("STEP 2: Generating Negative Samples")
    print("="*80)
    
    # Check if negatives already exist
    if os.path.exists(output_csv) and not force_regenerate:
        df_existing = pd.read_csv(output_csv)
        print(f"\n⚠ Negatives already exist at {output_csv}")
        print(f"  Found {len(df_existing)} negative samples")
        print("  Use --force-regenerate to regenerate.")
        return
    
    print(f"\nTrain metadata: {train_meta_path}")
    print(f"Image root (full training images): {img_root}")
    print(f"Output CSV: {output_csv}")
    
    print(f"\nLoading train metadata...")
    samples = load_train_meta(train_meta_path, img_root)
    print(f"Loaded {len(samples)} samples")
    
    print(f"Building tracks with bbox_mode={bbox_mode}...")
    tracks = build_tracks(samples)
    print(f"Built {len(tracks)} tracks")
    
    cleaned_samples = clean_tracks(tracks, mode=bbox_mode)
    print(f"Cleaned to {len(cleaned_samples)} samples")
    
    # Generate background negatives
    print(f"\nGenerating background negatives (num_bg_per_frame={num_bg_per_frame})...")
    bg_neg = generate_background_negatives(
        samples=cleaned_samples,
        num_bg_per_frame=num_bg_per_frame,
    )
    print(f"✓ Generated {len(bg_neg)} background negatives")
    
    # Generate partial negatives
    print(f"\nGenerating partial negatives (num_partial_per_box={num_partial_per_box})...")
    partial_neg = generate_partial_negatives(
        samples=cleaned_samples,
        num_partial_per_box=num_partial_per_box,
        max_iou_with_gt=max_iou_partial,
    )
    print(f"✓ Generated {len(partial_neg)} partial negatives")
    
    # Combine all negatives
    neg_samples = bg_neg + partial_neg
    print(f"\nTotal negative samples: {len(neg_samples)}")
    
    # Convert to DataFrame and save
    print(f"\nSaving to {output_csv}...")
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
    
    print("\n" + "="*80)
    print("Negative Generation Complete!")
    print("="*80)
    print(f"✓ Saved {len(df)} negative samples")
    print(f"  - Background negatives: {len(bg_neg)}")
    print(f"  - Partial negatives: {len(partial_neg)}")
    print(f"  - Unique quarters: {df['quarter'].nunique()}")
    print(f"  - Unique angles: {df['angle'].nunique()}")
    print(f"  - Total frames: {df['frame'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare all data before training (crops + negatives)"
    )
    
    # Paths
    parser.add_argument(
        "--train_meta",
        type=str,
        default="inputs/atmaCup22_2nd/train_meta.csv",
        help="Path to train_meta.csv"
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="inputs/atmaCup22_2nd/images",
        help="Root directory for full training images"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="inputs/train_crops_cache",
        help="Directory for cached crops"
    )
    parser.add_argument(
        "--negatives_output",
        type=str,
        default="inputs/train_negatives.csv",
        help="Output CSV file path for negatives"
    )
    
    # Skip options
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip crop cache preparation"
    )
    parser.add_argument(
        "--skip-negatives",
        action="store_true",
        help="Skip negative samples generation"
    )
    
    # Force options
    parser.add_argument(
        "--force-recrop",
        action="store_true",
        help="Force re-crop all images even if cache exists"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regenerate negatives even if they exist"
    )
    
    # Negative generation parameters
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
    
    # Crop cache parameters
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=1.2,
        help="Expand ratio for cropping"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for cached crops"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.negatives_output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)
    
    # Step 1: Prepare crop cache
    if not args.skip_cache:
        prepare_crop_cache(
            train_meta_path=args.train_meta,
            img_root=args.img_root,
            cache_dir=args.cache_dir,
            expand_ratio=args.expand_ratio,
            quality=args.quality,
            force_recrop=args.force_recrop,
        )
    else:
        print("\n⊘ Skipping crop cache preparation")
    
    # Step 2: Generate negatives
    if not args.skip_negatives:
        prepare_negatives(
            train_meta_path=args.train_meta,
            img_root=args.img_root,
            output_csv=args.negatives_output,
            num_bg_per_frame=args.num_bg_per_frame,
            num_partial_per_box=args.num_partial_per_box,
            max_iou_partial=args.max_iou_partial,
            bbox_mode="drop",
            force_regenerate=args.force_regenerate,
        )
    else:
        print("\n⊘ Skipping negative samples generation")
    
    print("\n" + "="*80)
    print("ALL DATA PREPARATION COMPLETE!")
    print("="*80)
    print("\nYou can now run training with:")
    print("  python run_cv_train.py")


if __name__ == "__main__":
    main()
