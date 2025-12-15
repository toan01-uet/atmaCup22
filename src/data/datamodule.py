from typing import Optional, Tuple, List
import logging
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.utils.utils import (
    load_train_meta,
    build_tracks,
    clean_tracks,
    split_labels_by_ratio,
    BBoxSample,
    make_label_folds
)
from src.data.dataset import PlayerReIDDataset
from src.utils.crop_cache import prepare_cropped_cache, get_cache_stats

class PlayerReIDDataModuleCV(pl.LightningDataModule):
    """
    DataModule cho cross-validation theo label:
      - n_folds fold
      - mỗi fold có unknown_per_fold label được loại khỏi training
    """

    def __init__(
        self,
        train_meta_path: str,
        img_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        bbox_mode: str = "drop",
        image_size: int = 224,
        n_folds: int = 5,
        fold_idx: int = 0,
        unknown_per_fold: int = 2,
        cv_seed: int = 42,
        logger: Optional[logging.Logger] = None,
        use_crop_cache: bool = True,
        cache_dir: Optional[str] = None,
        expand_ratio: float = 1.2,
    ):
        super().__init__()
        self.train_meta_path = train_meta_path
        self.img_root = img_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.bbox_mode = bbox_mode
        self.image_size = image_size
        self.logger_obj = logger
        self.use_crop_cache = use_crop_cache
        self.expand_ratio = expand_ratio

        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.unknown_per_fold = unknown_per_fold
        self.cv_seed = cv_seed

        # Set default cache directory
        if cache_dir is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.cache_dir = os.path.join(script_dir, "inputs", "train_crops_cache")
        else:
            self.cache_dir = cache_dir

        self.train_samples: Optional[List[BBoxSample]] = None
        self.val_samples: Optional[List[BBoxSample]] = None
        self.num_classes: Optional[int] = None
        self.unknown_labels_this_fold: Optional[List[int]] = None

    def setup(self, stage: Optional[str] = None):
        if self.logger_obj:
            self.logger_obj.info("Loading train metadata for CV...")
        # 1. load & clean
        samples = load_train_meta(self.train_meta_path, self.img_root)
        if self.logger_obj:
            self.logger_obj.info(f"Loaded {len(samples)} samples from metadata")
        
        if self.logger_obj:
            self.logger_obj.info(f"Building tracks with bbox_mode={self.bbox_mode}...")
        tracks = build_tracks(samples)
        if self.logger_obj:
            self.logger_obj.info(f"Built {len(tracks)} tracks")
        
        cleaned_samples = clean_tracks(tracks, mode=self.bbox_mode)
        if self.logger_obj:
            self.logger_obj.info(f"Cleaned to {len(cleaned_samples)} samples")

        # 2. unique labels
        all_labels = sorted({s.label_id for s in cleaned_samples})
        if self.logger_obj:
            self.logger_obj.info(f"Total unique labels: {len(all_labels)}")
            self.logger_obj.info(f"Creating {self.n_folds} folds with {self.unknown_per_fold} unknown labels per fold...")
        
        folds_unknown = make_label_folds(
            unique_labels=all_labels,
            n_folds=self.n_folds,
            unknown_per_fold=self.unknown_per_fold,
            seed=self.cv_seed,
        )

        # 3. unknown labels for this fold
        assert 0 <= self.fold_idx < self.n_folds
        unknown_labels = set(folds_unknown[self.fold_idx])
        self.unknown_labels_this_fold = list(unknown_labels)
        
        if self.logger_obj:
            self.logger_obj.info(f"Fold {self.fold_idx}: Unknown labels = {self.unknown_labels_this_fold}")

        # 4. known labels dùng để train/val
        known_labels = [lab for lab in all_labels if lab not in unknown_labels]
        if self.logger_obj:
            self.logger_obj.info(f"Known labels for training: {len(known_labels)}")

        # 5. split known_labels thành train_labels + val_labels (vẫn cần val monitor F1)
        if self.logger_obj:
            self.logger_obj.info(f"Splitting known labels with train_ratio={self.train_ratio}...")
        train_labels, val_labels = split_labels_by_ratio(
            known_labels, train_ratio=self.train_ratio
        )
        if self.logger_obj:
            self.logger_obj.info(f"Train labels: {len(train_labels)}, Val labels: {len(val_labels)}")

        # 6. Create label remapping to contiguous range [0, num_classes-1]
        all_used_labels = sorted(set(train_labels + val_labels))
        label_to_idx = {label: idx for idx, label in enumerate(all_used_labels)}
        
        if self.logger_obj:
            self.logger_obj.info(f"Created label mapping: {len(label_to_idx)} labels -> [0, {len(label_to_idx)-1}]")
        
        # 7. Remap label_ids in samples to contiguous indices
        def remap_sample(s: BBoxSample) -> BBoxSample:
            from dataclasses import replace
            return replace(s, label_id=label_to_idx[s.label_id])
        
        train_samples_raw = [s for s in cleaned_samples if s.label_id in train_labels]
        val_samples_raw = [s for s in cleaned_samples if s.label_id in val_labels]
        
        # 8. Prepare crop cache if enabled
        if self.use_crop_cache:
            if self.logger_obj:
                cache_stats = get_cache_stats(self.cache_dir)
                self.logger_obj.info(f"Crop cache: {cache_stats}")
                self.logger_obj.info(f"Preparing cropped cache at {self.cache_dir}...")
            
            # Cache training samples
            train_samples_cached = prepare_cropped_cache(
                train_samples_raw,
                self.cache_dir,
                expand_ratio=self.expand_ratio,
                quality=95
            )
            # Cache validation samples
            val_samples_cached = prepare_cropped_cache(
                val_samples_raw,
                self.cache_dir,
                expand_ratio=self.expand_ratio,
                quality=95
            )
            
            # Remap after caching
            self.train_samples = [remap_sample(s) for s in train_samples_cached]
            self.val_samples = [remap_sample(s) for s in val_samples_cached]
        else:
            # No caching, just remap
            self.train_samples = [remap_sample(s) for s in train_samples_raw]
            self.val_samples = [remap_sample(s) for s in val_samples_raw]
        
        self.num_classes = len(label_to_idx)  # số class cho ArcFace
        
        if self.logger_obj:
            self.logger_obj.info(f"Data split complete:")
            self.logger_obj.info(f"  - Train samples: {len(self.train_samples)}")
            self.logger_obj.info(f"  - Val samples: {len(self.val_samples)}")
            self.logger_obj.info(f"  - Number of classes: {self.num_classes}")
            self.logger_obj.info(f"  - Label range: [0, {self.num_classes-1}]")

    def _build_transforms(self, is_train: bool = True):
        if is_train:
            return T.Compose(
                [
                    T.Resize((self.image_size, self.image_size)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            return T.Compose(
                [
                    T.Resize((self.image_size, self.image_size)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def train_dataloader(self):
        assert self.train_samples is not None
        transform = self._build_transforms(is_train=True)
        ds = PlayerReIDDataset(
            self.train_samples, 
            transform=transform, 
            expand_ratio=self.expand_ratio,
            use_cache=self.use_crop_cache
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        assert self.val_samples is not None
        transform = self._build_transforms(is_train=False)
        ds = PlayerReIDDataset(
            self.val_samples, 
            transform=transform, 
            expand_ratio=self.expand_ratio,
            use_cache=self.use_crop_cache
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        
class PlayerReIDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_meta_path: str,
        img_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        area_multiplier: float = 2.0,
        bbox_mode: str = "drop",  # 'drop' hoặc 'interp'
        image_size: int = 224,
        logger: Optional[logging.Logger] = None,
        use_crop_cache: bool = True,
        cache_dir: Optional[str] = None,
        expand_ratio: float = 1.2,
    ):
        super().__init__()
        self.train_meta_path = train_meta_path
        self.img_root = img_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.area_multiplier = area_multiplier
        self.bbox_mode = bbox_mode
        self.image_size = image_size
        self.logger_obj = logger
        self.use_crop_cache = use_crop_cache
        self.expand_ratio = expand_ratio

        # Set default cache directory
        if cache_dir is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.cache_dir = os.path.join(script_dir, "inputs", "train_crops_cache")
        else:
            self.cache_dir = cache_dir

        self.train_samples: Optional[List[BBoxSample]] = None
        self.val_samples: Optional[List[BBoxSample]] = None
        self.num_classes: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        if self.logger_obj:
            self.logger_obj.info("Loading train metadata...")
        # 1. load & clean
        samples = load_train_meta(self.train_meta_path, self.img_root)
        if self.logger_obj:
            self.logger_obj.info(f"Loaded {len(samples)} samples from metadata")
        
        samples = [s for s in samples if s.angle == "side"] # train side only
        if self.logger_obj:
            self.logger_obj.info(f"Filtered to {len(samples)} side-angle samples")
        
        if self.logger_obj:
            self.logger_obj.info(f"Building tracks with bbox_mode={self.bbox_mode}...")
        tracks = build_tracks(samples)
        if self.logger_obj:
            self.logger_obj.info(f"Built {len(tracks)} tracks")
        
        cleaned_samples = clean_tracks(tracks, mode=self.bbox_mode)
        if self.logger_obj:
            self.logger_obj.info(f"Cleaned to {len(cleaned_samples)} samples")

        # 2. Split samples stratified by label (keep all labels in both train and val)
        from collections import defaultdict
        if self.logger_obj:
            self.logger_obj.info(f"Splitting samples with train_ratio={self.train_ratio}...")
        
        samples_by_label = defaultdict(list)
        for s in cleaned_samples:
            samples_by_label[s.label_id].append(s)
        
        train_samples = []
        val_samples = []
        for label_id, label_samples in samples_by_label.items():
            # Split each class' samples
            n_train = int(len(label_samples) * self.train_ratio)
            train_samples.extend(label_samples[:n_train])
            val_samples.extend(label_samples[n_train:])

        # Create label remapping to contiguous range [0, num_classes-1]
        all_labels = sorted(samples_by_label.keys())
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        
        if self.logger_obj:
            self.logger_obj.info(f"Created label mapping: {len(label_to_idx)} labels -> [0, {len(label_to_idx)-1}]")
        
        # Remap label_ids in samples to contiguous indices
        def remap_sample(s: BBoxSample) -> BBoxSample:
            from dataclasses import replace
            return replace(s, label_id=label_to_idx[s.label_id])
        
        # Prepare crop cache if enabled
        if self.use_crop_cache:
            if self.logger_obj:
                cache_stats = get_cache_stats(self.cache_dir)
                self.logger_obj.info(f"Crop cache: {cache_stats}")
                self.logger_obj.info(f"Preparing cropped cache at {self.cache_dir}...")
            
            # Cache training samples
            train_samples_cached = prepare_cropped_cache(
                train_samples,
                self.cache_dir,
                expand_ratio=self.expand_ratio,
                quality=95
            )
            # Cache validation samples
            val_samples_cached = prepare_cropped_cache(
                val_samples,
                self.cache_dir,
                expand_ratio=self.expand_ratio,
                quality=95
            )
            
            # Remap after caching
            self.train_samples = [remap_sample(s) for s in train_samples_cached]
            self.val_samples = [remap_sample(s) for s in val_samples_cached]
        else:
            # No caching, just remap
            self.train_samples = [remap_sample(s) for s in train_samples]
            self.val_samples = [remap_sample(s) for s in val_samples]
        
        self.num_classes = len(samples_by_label)  # All unique labels
        
        if self.logger_obj:
            self.logger_obj.info(f"Data split complete:")
            self.logger_obj.info(f"  - Train samples: {len(self.train_samples)}")
            self.logger_obj.info(f"  - Val samples: {len(self.val_samples)}")
            self.logger_obj.info(f"  - Number of classes: {self.num_classes}")
            self.logger_obj.info(f"  - Label range: [0, {self.num_classes-1}]")

    def _build_transforms(self, is_train: bool = True):
        if is_train:
            return T.Compose(
                [
                    T.Resize((self.image_size, self.image_size)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            return T.Compose(
                [
                    T.Resize((self.image_size, self.image_size)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def train_dataloader(self):
        assert self.train_samples is not None
        transform = self._build_transforms(is_train=True)
        ds = PlayerReIDDataset(
            self.train_samples, 
            transform=transform, 
            expand_ratio=self.expand_ratio,
            use_cache=self.use_crop_cache
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        assert self.val_samples is not None
        transform = self._build_transforms(is_train=False)
        ds = PlayerReIDDataset(
            self.val_samples, 
            transform=transform, 
            expand_ratio=self.expand_ratio,
            use_cache=self.use_crop_cache
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

