from typing import Optional, Tuple, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from utils import (
    load_train_meta,
    build_tracks,
    clean_tracks,
    split_labels_by_ratio,
    BBoxSample,
)
from dataset import PlayerReIDDataset


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

        self.train_samples: Optional[List[BBoxSample]] = None
        self.val_samples: Optional[List[BBoxSample]] = None
        self.num_classes: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        # 1. load & clean
        samples = load_train_meta(self.train_meta_path, self.img_root)
        samples = [s for s in samples if s.angle == "side"] # train side only
        
        tracks = build_tracks(samples)
        cleaned_samples = clean_tracks(tracks, mode=self.bbox_mode)

        # 2. split theo label
        all_labels = [s.label_id for s in cleaned_samples]
        train_labels, val_labels = split_labels_by_ratio(
            all_labels, train_ratio=self.train_ratio
        )

        train_samples = [s for s in cleaned_samples if s.label_id in train_labels]
        val_samples = [s for s in cleaned_samples if s.label_id in val_labels]

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.num_classes = len(set(train_labels))  # dùng cho model

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
        ds = PlayerReIDDataset(self.train_samples, transform=transform, expand_ratio=1.2)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        assert self.val_samples is not None
        transform = self._build_transforms(is_train=False)
        ds = PlayerReIDDataset(self.val_samples, transform=transform, expand_ratio=1.2)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
