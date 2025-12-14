"""
Script to pre-crop and cache all training images.
Run this before training to significantly speed up data loading.
"""

import os
import sys
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.utils.utils import load_train_meta, build_tracks, clean_tracks
from src.utils.crop_cache import prepare_cropped_cache, get_cache_stats

if __name__ == "__main__":
    print("="*80)
    print("Pre-cropping Training Images")
    print("="*80)
    
    # Paths
    train_meta_path = os.path.join(script_dir, "inputs", "atmaCup22_2nd_meta", "train_meta.csv")
    img_root = os.path.join(script_dir, "inputs", "images")
    cache_dir = os.path.join(script_dir, "inputs", "train_crops_cache")
    
    print(f"\nTrain metadata: {train_meta_path}")
    print(f"Image root: {img_root}")
    print(f"Cache directory: {cache_dir}")
    
    # Check current cache status
    print("\nChecking cache status...")
    cache_stats = get_cache_stats(cache_dir)
    print(f"Cache exists: {cache_stats['exists']}")
    print(f"Cached files: {cache_stats['num_files']}")
    print(f"Total size: {cache_stats['total_size_mb']:.2f} MB")
    
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
    
    # Pre-crop with expand_ratio=1.2 (default used in training)
    print("\n" + "="*80)
    print("Starting cropping process...")
    print("="*80)
    
    expand_ratio = 1.2
    quality = 95
    
    cached_samples = prepare_cropped_cache(
        cleaned_samples,
        cache_dir,
        expand_ratio=expand_ratio,
        quality=quality,
        force_recrop=False  # Set to True to re-crop all
    )
    
    # Final cache stats
    print("\n" + "="*80)
    print("Cropping Complete!")
    print("="*80)
    final_stats = get_cache_stats(cache_dir)
    print(f"Cached files: {final_stats['num_files']}")
    print(f"Total size: {final_stats['total_size_mb']:.2f} MB")
    print(f"Average size per crop: {final_stats['total_size_mb']/final_stats['num_files']:.3f} MB")
    print("\nCache is ready for training!")
