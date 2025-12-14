"""
Utility functions for pre-cropping and caching training images.
This significantly speeds up data loading by avoiding repeated cropping operations.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional
from PIL import Image
from tqdm import tqdm

from src.utils.utils import BBoxSample


def get_crop_cache_path(
    sample: BBoxSample,
    cache_dir: str,
    expand_ratio: float = 1.0
) -> str:
    """
    Generate a unique cache path for a cropped image based on sample metadata.
    Uses hash to create unique filename.
    """
    # Create unique identifier from sample data
    identifier = f"{sample.quarter}_{sample.angle}_{sample.session}_{sample.frame}_{sample.x}_{sample.y}_{sample.w}_{sample.h}_{sample.label_id}_{expand_ratio}"
    hash_id = hashlib.md5(identifier.encode()).hexdigest()
    
    # Organize by quarter and angle for better file system organization
    subdir = os.path.join(cache_dir, f"{sample.quarter}", f"{sample.angle}")
    os.makedirs(subdir, exist_ok=True)
    
    return os.path.join(subdir, f"{hash_id}.jpg")


def crop_and_cache_image(
    sample: BBoxSample,
    cache_path: str,
    expand_ratio: float = 1.0,
    quality: int = 95
) -> None:
    """
    Crop an image according to bbox and save to cache.
    """
    img = Image.open(sample.img_path).convert("RGB")
    w_img, h_img = img.size
    x, y, w, h = sample.x, sample.y, sample.w, sample.h

    if expand_ratio != 1.0:
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_w = w * expand_ratio
        new_h = h * expand_ratio
        x = int(cx - new_w / 2.0)
        y = int(cy - new_h / 2.0)
        w = int(new_w)
        h = int(new_h)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    crop = img.crop((x1, y1, x2, y2))
    crop.save(cache_path, "JPEG", quality=quality)


def prepare_cropped_cache(
    samples: List[BBoxSample],
    cache_dir: str,
    expand_ratio: float = 1.0,
    quality: int = 95,
    force_recrop: bool = False
) -> List[BBoxSample]:
    """
    Pre-crop and cache all training images. Returns updated samples with cached paths.
    
    Args:
        samples: List of BBoxSample to process
        cache_dir: Directory to store cropped images
        expand_ratio: Expansion ratio for bounding boxes
        quality: JPEG quality for saved crops
        force_recrop: If True, re-crop even if cache exists
        
    Returns:
        List of BBoxSample with img_path updated to cached crop paths
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cached_samples = []
    to_process = []
    
    # First pass: check which samples need processing
    for sample in samples:
        cache_path = get_crop_cache_path(sample, cache_dir, expand_ratio)
        
        if os.path.exists(cache_path) and not force_recrop:
            # Cache exists, just update path
            from dataclasses import replace
            cached_sample = replace(sample, img_path=cache_path)
            cached_samples.append(cached_sample)
        else:
            # Need to crop
            to_process.append((sample, cache_path))
    
    # Second pass: crop images that need processing
    if to_process:
        print(f"Cropping {len(to_process)} images to cache...")
        for sample, cache_path in tqdm(to_process, desc="Caching crops"):
            try:
                crop_and_cache_image(sample, cache_path, expand_ratio, quality)
                from dataclasses import replace
                cached_sample = replace(sample, img_path=cache_path)
                cached_samples.append(cached_sample)
            except Exception as e:
                print(f"Error cropping {sample.img_path}: {e}")
                # Keep original path on error
                cached_samples.append(sample)
    
    print(f"Cache ready: {len(cached_samples)} samples ({len(to_process)} newly cropped)")
    return cached_samples


def get_cache_stats(cache_dir: str) -> dict:
    """
    Get statistics about the crop cache.
    """
    if not os.path.exists(cache_dir):
        return {
            "exists": False,
            "num_files": 0,
            "total_size_mb": 0
        }
    
    num_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.jpg'):
                num_files += 1
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    return {
        "exists": True,
        "num_files": num_files,
        "total_size_mb": total_size / (1024 * 1024)
    }
