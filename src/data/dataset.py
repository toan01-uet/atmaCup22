from typing import List, Callable, Optional, Tuple
import os

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.utils.utils import BBoxSample

class PlayerReIDDataset(Dataset):
    """
    Dataset cơ bản: crop bbox -> image tensor, trả về (image, label_id).
    Supports both pre-cropped cache and on-the-fly cropping.
    """

    def __init__(
        self,
        samples: List[BBoxSample],
        transform: Optional[Callable] = None,
        expand_ratio: float = 1.0,
        use_cache: bool = True,
    ):
        """
        Args:
            samples: List of BBoxSample (can point to cached crops or original images)
            transform: Transforms to apply
            expand_ratio: Expansion ratio for bbox (only used if not pre-cached)
            use_cache: Whether samples are already pointing to cached crops
        """
        self.samples = samples
        self.transform = transform
        self.expand_ratio = expand_ratio
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, img: Image.Image, s: BBoxSample) -> Image.Image:
        """Crop image on-the-fly (only used if not using cache)"""
        w_img, h_img = img.size
        x, y, w, h = s.x, s.y, s.w, s.h

        if self.expand_ratio != 1.0:
            cx = x + w / 2.0
            cy = y + h / 2.0
            new_w = w * self.expand_ratio
            new_h = h * self.expand_ratio
            x = int(cx - new_w / 2.0)
            y = int(cy - new_h / 2.0)
            w = int(new_w)
            h = int(new_h)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        return img.crop((x1, y1, x2, y2))

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        
        # Check if img_path points to a cached crop (in crops directory)
        # or if it needs on-the-fly cropping
        if self.use_cache or 'train_crops_cache' in s.img_path or s.img_path.endswith('.jpg') and os.path.dirname(s.img_path).endswith(('crops', 'crops_cache')):
            # Already cropped, just load
            crop = Image.open(s.img_path).convert("RGB")
        else:
            # Need to crop on-the-fly
            img = Image.open(s.img_path).convert("RGB")
            crop = self._crop(img, s)
        
        if self.transform is not None:
            crop = self.transform(crop)
        label = s.label_id
        return crop, torch.tensor(label, dtype=torch.long)


class MultiViewPlayerReIDDataset(Dataset):
    """
    Dataset cho multi-view: mỗi sample = (img_side, img_top, label_id).
    Dùng cho training consistency loss giữa 2 góc nhìn.
    """

    def __init__(
        self,
        pairs: List[Tuple[BBoxSample, BBoxSample]],
        transform: Optional[Callable] = None,
        expand_ratio: float = 1.0,
    ):
        self.pairs = pairs
        self.transform = transform
        self.expand_ratio = expand_ratio

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(self, img: Image.Image, s: BBoxSample) -> Image.Image:
        w_img, h_img = img.size
        x, y, w, h = s.x, s.y, s.w, s.h

        if self.expand_ratio != 1.0:
            cx = x + w / 2.0
            cy = y + h / 2.0
            new_w = w * self.expand_ratio
            new_h = h * self.expand_ratio
            x = int(cx - new_w / 2.0)
            y = int(cy - new_h / 2.0)
            w = int(new_w)
            h = int(new_h)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        return img.crop((x1, y1, x2, y2))

    def __getitem__(self, idx: int):
        side_s, top_s = self.pairs[idx]
        label = side_s.label_id  # = top_s.label_id

        img_side = Image.open(side_s.img_path).convert("RGB")
        img_top = Image.open(top_s.img_path).convert("RGB")

        crop_side = self._crop(img_side, side_s)
        crop_top = self._crop(img_top, top_s)

        if self.transform is not None:
            crop_side = self.transform(crop_side)
            crop_top = self.transform(crop_top)

        return crop_side, crop_top, torch.tensor(label, dtype=torch.long)
