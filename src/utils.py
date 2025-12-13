import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

import numpy as np
import pandas as pd


@dataclass
class BBoxSample:
    quarter: str
    angle: str
    session: int
    frame: int
    x: int
    y: int
    w: int
    h: int
    label_id: int
    img_path: str


def load_train_meta(csv_path: str, img_root: str) -> List[BBoxSample]:
    """
    Đọc train_meta.csv và map sang list[BBoxSample].
    img_root: thư mục chứa ảnh gốc.
    """
    df = pd.read_csv(csv_path)
    samples: List[BBoxSample] = []

    for _, row in df.iterrows():
        quarter = str(row["quarter"])
        angle = str(row["angle"])
        session = int(row["session"])
        frame = int(row["frame"])
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["w"])
        h = int(row["h"])
        label_id = int(row["label_id"])

        img_name = f"{quarter}__{angle}__{session}__{frame}.jpg"
        img_path = os.path.join(img_root, img_name)

        samples.append(
            BBoxSample(
                quarter=quarter,
                angle=angle,
                session=session,
                frame=frame,
                x=x,
                y=y,
                w=w,
                h=h,
                label_id=label_id,
                img_path=img_path,
            )
        )

    return samples


def build_tracks(samples: List[BBoxSample]) -> Dict[Tuple[str, str, int], List[BBoxSample]]:
    """
    Group theo (quarter, angle, label_id) và sort theo (session, frame).
    """
    from collections import defaultdict

    tracks: Dict[Tuple[str, str, int], List[BBoxSample]] = defaultdict(list)
    for s in samples:
        key = (s.quarter, s.angle, s.label_id)
        tracks[key].append(s)

    for key in tracks:
        tracks[key].sort(key=lambda s: (s.session, s.frame))

    return tracks


def detect_outliers_in_track(track: List[BBoxSample],
                             area_multiplier: float = 2.0) -> List[bool]:
    """
    Flag những bbox có area quá lớn so với median của track.
    Trả về list bool cùng chiều track.
    """
    areas = np.array([s.w * s.h for s in track], dtype=np.float32)
    median_area = np.median(areas)
    if median_area <= 0:
        return [False] * len(track)

    flags: List[bool] = []
    for area in areas:
        if area > area_multiplier * median_area:
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _next_valid_prev(track: List[BBoxSample],
                     flags: List[bool],
                     idx: int) -> Optional[BBoxSample]:
    for j in range(idx - 1, -1, -1):
        if not flags[j]:
            return track[j]
    return None


def _next_valid_next(track: List[BBoxSample],
                     flags: List[bool],
                     idx: int) -> Optional[BBoxSample]:
    for j in range(idx + 1, len(track)):
        if not flags[j]:
            return track[j]
    return None


def _copy_sample(s: BBoxSample) -> BBoxSample:
    return BBoxSample(
        quarter=s.quarter,
        angle=s.angle,
        session=s.session,
        frame=s.frame,
        x=s.x,
        y=s.y,
        w=s.w,
        h=s.h,
        label_id=s.label_id,
        img_path=s.img_path,
    )


def clean_tracks(tracks: Dict[Tuple[str, str, int], List[BBoxSample]],
                 mode: str = "drop") -> List[BBoxSample]:
    """
    Làm sạch bbox outlier trong mỗi track.
    mode = 'drop'  -> bỏ frame outlier
    mode = 'interp'-> nội suy bbox từ frame lân cận (nếu có)
    """
    cleaned: List[BBoxSample] = []

    for key, track in tracks.items():
        flags = detect_outliers_in_track(track)
        if not any(flags):
            cleaned.extend(track)
            continue

        n = len(track)
        for i, (s, is_outlier) in enumerate(zip(track, flags)):
            if not is_outlier:
                cleaned.append(s)
                continue

            if mode == "drop":
                continue
            elif mode == "interp":
                prev = _next_valid_prev(track, flags, i)
                nxt = _next_valid_next(track, flags, i)
                if prev is None or nxt is None:
                    # thiếu context, bỏ luôn
                    continue
                new_s = _copy_sample(s)
                new_s.x = int((prev.x + nxt.x) / 2)
                new_s.y = int((prev.y + nxt.y) / 2)
                new_s.w = int((prev.w + nxt.w) / 2)
                new_s.h = int((prev.h + nxt.h) / 2)
                cleaned.append(new_s)

    return cleaned


def split_labels_by_ratio(label_ids: List[int],
                          train_ratio: float = 0.8,
                          seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Chia list label thành 2 tập (train_labels, val_labels) theo tỉ lệ.
    Dùng để tách train/val theo player, tránh leak theo frame.
    """
    unique_labels = sorted(set(label_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_labels)
    n_train = int(len(unique_labels) * train_ratio)
    train_labels = unique_labels[:n_train]
    val_labels = unique_labels[n_train:]
    return train_labels, val_labels


# ========== OPEN-SET SPLIT ==========

def split_labels_open_set(
    label_ids: List[int],
    unknown_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Chia tập label thành:
      - known_labels: dùng để train + build gallery
      - unknown_labels: chỉ dùng để tune threshold (coi như "người lạ")
    unknown_ratio: tỷ lệ label cho vào unknown.
    """
    unique_labels = sorted(set(label_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_labels)
    n_unknown = int(len(unique_labels) * unknown_ratio)
    unknown_labels = unique_labels[:n_unknown]
    known_labels = unique_labels[n_unknown:]
    return known_labels, unknown_labels


# ========== MULTI-VIEW PAIRS (side + top) ==========

def build_multiview_pairs(samples: List[BBoxSample]) -> List[Tuple[BBoxSample, BBoxSample]]:
    """
    Tạo danh sách (side_sample, top_sample) cho những (quarter, session, frame, label_id)
    có đủ cả 2 góc.
    """
    from collections import defaultdict

    groups: Dict[Tuple[str, int, int, int], Dict[str, List[BBoxSample]]] = defaultdict(
        lambda: {"side": [], "top": []}
    )

    for s in samples:
        key = (s.quarter, s.session, s.frame, s.label_id)
        groups[key][s.angle].append(s)

    pairs: List[Tuple[BBoxSample, BBoxSample]] = []
    for key, views in groups.items():
        side_list = views["side"]
        top_list = views["top"]
        if not side_list or not top_list:
            continue
        # lấy 1 mẫu side và 1 mẫu top đầu tiên (có thể random nếu muốn)
        side_sample = side_list[0]
        top_sample = top_list[0]
        pairs.append((side_sample, top_sample))

    return pairs


# ========== TEST BBOX & TRACKLET (cho inference) ==========

@dataclass
class TestBBox:
    idx: int  # index trong test_meta.csv
    quarter: str
    angle: str
    session: int
    frame: int
    x: int
    y: int
    w: int
    h: int
    img_path: str


@dataclass
class Tracklet:
    bboxes: List[TestBBox]
    pred_label: Optional[int] = None
    pred_score: Optional[float] = None  # similarity score (cosine)


def load_test_meta(csv_path: str, img_root: str) -> List[TestBBox]:
    df = pd.read_csv(csv_path)
    bboxes: List[TestBBox] = []
    for idx, row in df.iterrows():
        quarter = str(row["quarter"])
        angle = str(row["angle"])
        session = int(row["session"])
        frame = int(row["frame"])
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["w"])
        h = int(row["h"])

        img_name = f"{quarter}__{angle}__{session}__{frame}.jpg"
        img_path = os.path.join(img_root, img_name)

        bboxes.append(
            TestBBox(
                idx=idx,
                quarter=quarter,
                angle=angle,
                session=session,
                frame=frame,
                x=x,
                y=y,
                w=w,
                h=h,
                img_path=img_path,
            )
        )
    return bboxes


def iou_bbox(a: TestBBox, b: TestBBox) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a.w * a.h
    area_b = b.w * b.h
    union_area = area_a + area_b - inter_area + 1e-6

    return inter_area / union_area


def build_test_tracklets(
    test_bboxes: List[TestBBox],
    max_frame_gap: int = 2,
    iou_threshold: float = 0.3,
) -> List[Tracklet]:
    """
    IOU-based tracker đơn giản để ghép tracklet trong test theo (quarter, angle, session).
    """
    from collections import defaultdict

    groups: Dict[Tuple[str, str, int], List[TestBBox]] = defaultdict(list)
    for b in test_bboxes:
        key = (b.quarter, b.angle, b.session)
        groups[key].append(b)

    all_tracklets: List[Tracklet] = []

    for key, bboxes in groups.items():
        bboxes.sort(key=lambda b: b.frame)
        tracklets: List[Tracklet] = []

        for b in bboxes:
            assigned = False
            for tr in tracklets:
                last = tr.bboxes[-1]
                if b.frame <= last.frame:
                    continue
                if b.frame - last.frame > max_frame_gap:
                    continue
                if iou_bbox(b, last) > iou_threshold:
                    tr.bboxes.append(b)
                    assigned = True
                    break
            if not assigned:
                new_tr = Tracklet(bboxes=[b])
                tracklets.append(new_tr)

        all_tracklets.extend(tracklets)

    return all_tracklets


# ========== GALLERY & EMBEDDING HELPER ==========

import torch
from PIL import Image
from torchvision import transforms as T


def default_infer_transform(image_size: int = 224):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [D], b: [D]
    """
    a = a / (a.norm(p=2) + 1e-6)
    b = b / (b.norm(p=2) + 1e-6)
    return (a * b).sum()


def build_gallery(
    pl_module,  # PlayerReIDModule
    samples: List[BBoxSample],
    batch_size: int = 64,
    image_size: int = 224,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Tính centroid embedding cho mỗi label_id từ list samples.
    """
    from collections import defaultdict

    pl_module.eval()
    pl_module.to(device)
    transform = default_infer_transform(image_size=image_size)

    label_to_embs: Dict[int, List[torch.Tensor]] = defaultdict(list)

    # mini batching
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        imgs = []
        labels = []
        for s in batch_samples:
            img = Image.open(s.img_path).convert("RGB")
            w_img, h_img = img.size
            x1 = max(0, s.x)
            y1 = max(0, s.y)
            x2 = min(w_img, s.x + s.w)
            y2 = min(h_img, s.y + s.h)
            crop = img.crop((x1, y1, x2, y2))
            crop = transform(crop)
            imgs.append(crop)
            labels.append(s.label_id)

        if not imgs:
            continue

        imgs_tensor = torch.stack(imgs, dim=0).to(device)
        with torch.no_grad():
            emb, _ = pl_module(imgs_tensor, labels=None)  # forward: (emb, logits)

        for e, lab in zip(emb, labels):
            label_to_embs[lab].append(e.cpu())

    gallery: Dict[int, torch.Tensor] = {}
    for lab, emb_list in label_to_embs.items():
        embs_stack = torch.stack(emb_list, dim=0)
        gallery[lab] = embs_stack.mean(dim=0)

    return gallery


def get_tracklet_embedding(
    pl_module,
    tracklet: Tracklet,
    image_size: int = 224,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Lấy embedding trung bình cho 1 tracklet.
    """
    pl_module.eval()
    pl_module.to(device)
    transform = default_infer_transform(image_size=image_size)

    embs: List[torch.Tensor] = []

    with torch.no_grad():
        for b in tracklet.bboxes:
            img = Image.open(b.img_path).convert("RGB")
            w_img, h_img = img.size
            x1 = max(0, b.x)
            y1 = max(0, b.y)
            x2 = min(w_img, b.x + b.w)
            y2 = min(h_img, b.y + b.h)
            crop = img.crop((x1, y1, x2, y2))
            crop = transform(crop).unsqueeze(0).to(device)
            emb, _ = pl_module(crop, labels=None)
            embs.append(emb.squeeze(0).cpu())

    if not embs:
        return torch.zeros(pl_module.hparams.embedding_dim)

    embs_stack = torch.stack(embs, dim=0)
    return embs_stack.mean(dim=0)
