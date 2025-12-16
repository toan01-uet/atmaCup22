import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
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

        img_name = f"{quarter}__{angle}__{session:02d}__{frame:02d}.jpg"
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


def split_samples_by_ratio_per_label(
    samples: List[BBoxSample],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[BBoxSample], List[BBoxSample]]:
    """
    Split samples into train/val while maintaining representation from each label.
    For each label, split its samples according to train_ratio.
    This ensures both train and val sets contain all labels.
    """
    from collections import defaultdict
    
    samples_by_label = defaultdict(list)
    for s in samples:
        samples_by_label[s.label_id].append(s)
    
    train_samples = []
    val_samples = []
    
    rng = random.Random(seed)
    for label_id, label_samples in samples_by_label.items():
        # Shuffle samples for this label
        shuffled = label_samples.copy()
        rng.shuffle(shuffled)
        
        n_train = int(len(shuffled) * train_ratio)
        train_samples.extend(shuffled[:n_train])
        val_samples.extend(shuffled[n_train:])
    
    return train_samples, val_samples


def save_fold_metadata(
    fold_idx: int,
    train_samples: List[BBoxSample],
    val_samples: List[BBoxSample],
    unknown_labels: List[int],
    output_dir: str
) -> None:
    """
    Save fold metadata (train/val samples and unknown labels) to CSV files.
    This ensures consistency between training and inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train samples
    train_df = pd.DataFrame([
        {
            'quarter': s.quarter,
            'angle': s.angle,
            'session': s.session,
            'frame': s.frame,
            'x': s.x,
            'y': s.y,
            'w': s.w,
            'h': s.h,
            'label_id': s.label_id,
            'img_path': s.img_path
        }
        for s in train_samples
    ])
    train_df.to_csv(os.path.join(output_dir, f'fold{fold_idx}_train_samples.csv'), index=False)
    
    # Save val samples
    val_df = pd.DataFrame([
        {
            'quarter': s.quarter,
            'angle': s.angle,
            'session': s.session,
            'frame': s.frame,
            'x': s.x,
            'y': s.y,
            'w': s.w,
            'h': s.h,
            'label_id': s.label_id,
            'img_path': s.img_path
        }
        for s in val_samples
    ])
    val_df.to_csv(os.path.join(output_dir, f'fold{fold_idx}_val_samples.csv'), index=False)
    
    # Save unknown labels
    unknown_df = pd.DataFrame({'label_id': unknown_labels})
    unknown_df.to_csv(os.path.join(output_dir, f'fold{fold_idx}_unknown_labels.csv'), index=False)


def load_fold_metadata(
    fold_idx: int,
    output_dir: str
) -> Tuple[List[BBoxSample], List[BBoxSample], List[int]]:
    """
    Load fold metadata from saved CSV files.
    """
    # Load train samples
    train_df = pd.read_csv(os.path.join(output_dir, f'fold{fold_idx}_train_samples.csv'))
    train_samples = [
        BBoxSample(
            quarter=str(row['quarter']),
            angle=str(row['angle']),
            session=int(row['session']),
            frame=int(row['frame']),
            x=int(row['x']),
            y=int(row['y']),
            w=int(row['w']),
            h=int(row['h']),
            label_id=int(row['label_id']),
            img_path=str(row['img_path'])
        )
        for _, row in train_df.iterrows()
    ]
    
    # Load val samples
    val_df = pd.read_csv(os.path.join(output_dir, f'fold{fold_idx}_val_samples.csv'))
    val_samples = [
        BBoxSample(
            quarter=str(row['quarter']),
            angle=str(row['angle']),
            session=int(row['session']),
            frame=int(row['frame']),
            x=int(row['x']),
            y=int(row['y']),
            w=int(row['w']),
            h=int(row['h']),
            label_id=int(row['label_id']),
            img_path=str(row['img_path'])
        )
        for _, row in val_df.iterrows()
    ]
    
    # Load unknown labels
    unknown_df = pd.read_csv(os.path.join(output_dir, f'fold{fold_idx}_unknown_labels.csv'))
    unknown_labels = unknown_df['label_id'].tolist()
    
    return train_samples, val_samples, unknown_labels


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
    """
    Load test metadata. New format has pre-cropped images with rel_path column.
    """
    df = pd.read_csv(csv_path)
    bboxes: List[TestBBox] = []
    
    # Check if this is the new format with rel_path (pre-cropped images)
    if 'rel_path' in df.columns:
        # New format: cropped images with rel_path
        for idx, row in df.iterrows():
            quarter = str(row["quarter"])
            angle = str(row["angle"])
            session = int(row["session_no"])
            frame = int(row["frame_in_session"])
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])
            
            # Use rel_path which points directly to cropped image
            img_path = os.path.join(img_root, row["rel_path"])
            
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
    else:
        # Old format: full images that need to be cropped
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

def group_samples_by_frame(samples: List[BBoxSample]):
    """
    Group train samples theo (quarter, angle, session, frame).
    """
    groups: Dict[Tuple[str, str, int, int], List[BBoxSample]] = defaultdict(list)
    for s in samples:
        key = (s.quarter, s.angle, s.session, s.frame)
        groups[key].append(s)
    return groups


def random_background_boxes_for_frame(
    img: Image.Image,
    frame_boxes: List[BBoxSample],
    num_bg: int = 2,
    min_size_ratio: float = 0.1,
    max_size_ratio: float = 0.4,
    iou_threshold: float = 0.1,
) -> List[Tuple[int, int, int, int]]:
    """
    Sinh một số bbox background không overlap nhiều với bbox cầu thủ.
    Trả về list (x, y, w, h). Không check label ở đây.
    """
    w_img, h_img = img.size
    gt_boxes = [(s.x, s.y, s.w, s.h) for s in frame_boxes]
    bg_boxes: List[Tuple[int, int, int, int]] = []

    trials = 0
    max_trials = num_bg * 20

    while len(bg_boxes) < num_bg and trials < max_trials:
        trials += 1
        box_w = random.randint(int(w_img * min_size_ratio), int(w_img * max_size_ratio))
        box_h = random.randint(int(h_img * min_size_ratio), int(h_img * max_size_ratio))
        x = random.randint(0, max(0, w_img - box_w))
        y = random.randint(0, max(0, h_img - box_h))

        candidate = (x, y, box_w, box_h)
        iou_with_any_gt = max(_iou_box(candidate, gt) for gt in gt_boxes) if gt_boxes else 0.0
        if iou_with_any_gt < iou_threshold:
            bg_boxes.append(candidate)

    return bg_boxes


def generate_background_negatives(
    samples: List[BBoxSample],
    num_bg_per_frame: int = 1,
) -> List[BBoxSample]:
    """
    Tạo negative background BBoxSample (label_id = -1) từ train.
    """
    groups = group_samples_by_frame(samples)
    background_samples: List[BBoxSample] = []

    for key, frame_boxes in groups.items():
        quarter, angle, session, frame = key
        img_path = frame_boxes[0].img_path  # cùng frame, cùng path
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        bg_boxes = random_background_boxes_for_frame(img, frame_boxes, num_bg=num_bg_per_frame)
        for x, y, w, h in bg_boxes:
            background_samples.append(
                BBoxSample(
                    quarter=quarter,
                    angle=angle,
                    session=session,
                    frame=frame,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    label_id=-1,  # unknown
                    img_path=img_path,
                )
            )
    return background_samples

def _iou_box(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = w1 * h1
        area_b = w2 * h2
        union = area_a + area_b - inter_area + 1e-6
        return inter_area / union

def generate_partial_negatives(
    samples: List[BBoxSample],
    num_partial_per_box: int = 1,
    max_iou_with_gt: float = 0.4,
) -> List[BBoxSample]:
    """
    Tạo các bbox "partial player": IoU < max_iou_with_gt so với bbox GT ban đầu.
    """
    partial_samples: List[BBoxSample] = []

    for s in samples:
        if s.w <= 0 or s.h <= 0:
            continue

        base_box = (s.x, s.y, s.w, s.h)

        for _ in range(num_partial_per_box):
            # shrink từ 0.3–0.7 kích thước
            shrink_ratio = random.uniform(0.3, 0.7)
            new_w = int(s.w * shrink_ratio)
            new_h = int(s.h * shrink_ratio)
            if new_w <= 0 or new_h <= 0:
                continue

            # random chọn vị trí trong bbox gốc
            max_dx = max(0, s.w - new_w)
            max_dy = max(0, s.h - new_h)
            dx = random.randint(0, max_dx)
            dy = random.randint(0, max_dy)
            x_new = s.x + dx
            y_new = s.y + dy

            partial_box = (x_new, y_new, new_w, new_h)
            if _iou_box(base_box, partial_box) < max_iou_with_gt:
                partial_samples.append(
                    BBoxSample(
                        quarter=s.quarter,
                        angle=s.angle,
                        session=s.session,
                        frame=s.frame,
                        x=x_new,
                        y=y_new,
                        w=new_w,
                        h=new_h,
                        label_id=-1,  # unknown
                        img_path=s.img_path,
                    )
                )
                # nếu muốn chỉ 1 partial / box:
                # break

    return partial_samples

def make_label_folds(
    unique_labels: List[int],
    n_folds: int = 5,
    unknown_per_fold: int = 2,
    seed: int = 42,
) -> List[List[int]]:
    """
    Chia danh sách label thành các nhóm unknown theo fold.
    Mỗi fold có đúng unknown_per_fold label làm unknown.

    Giả định: n_folds * unknown_per_fold <= len(unique_labels).
    """
    assert n_folds * unknown_per_fold <= len(unique_labels), \
        "Không đủ label để chia mỗi fold có unknown_per_fold labels."

    labels = list(unique_labels)
    rng = random.Random(seed)
    rng.shuffle(labels)

    folds_unknown: List[List[int]] = []
    for i in range(n_folds):
        start = i * unknown_per_fold
        end = start + unknown_per_fold
        folds_unknown.append(labels[start:end])  # 2 label / fold

    return folds_unknown


def load_negatives_from_csv(neg_csv_path: str) -> List[BBoxSample]:
    """
    Load pre-computed negative samples from CSV file.
    This is much faster than generating negatives on-the-fly during training.
    
    Args:
        neg_csv_path: Path to CSV file containing negative samples
        
    Returns:
        List of BBoxSample with label_id=-1 (unknown/negative)
        
    Usage:
        # After running 00_prepare_negatives.py:
        negatives = load_negatives_from_csv("inputs/train_negatives.csv")
        unknown_samples.extend(negatives)
    """
    if not os.path.exists(neg_csv_path):
        raise FileNotFoundError(
            f"Negatives CSV not found: {neg_csv_path}\n"
            f"Please run: python 00_prepare_negatives.py first"
        )
    
    df = pd.read_csv(neg_csv_path)
    samples = []
    
    for _, row in df.iterrows():
        samples.append(
            BBoxSample(
                quarter=str(row["quarter"]),
                angle=str(row["angle"]),
                session=int(row["session"]),
                frame=int(row["frame"]),
                x=int(row["x"]),
                y=int(row["y"]),
                w=int(row["w"]),
                h=int(row["h"]),
                label_id=int(row["label_id"]),  # Should be -1
                img_path=str(row["img_path"]),
            )
        )
    
    return samples
