# inference.py

from typing import List, Dict, Tuple
import torch
import numpy as np
from tqdm import tqdm

from utils import (
    BBoxSample,
    TestBBox,
    Tracklet,
    load_train_meta,
    build_tracks,
    clean_tracks,
    split_labels_open_set,
    load_test_meta,
    build_test_tracklets,
    build_gallery,
    get_tracklet_embedding,
    compute_cosine_similarity,
)
from pl_module import PlayerReIDModule


def build_open_set_splits(
    train_meta_path: str,
    img_root: str,
    bbox_mode: str = "drop",
    unknown_ratio: float = 0.2,
):
    """
    Đọc train_meta -> clean -> split label thành known/unknown cho open-set.
    Trả về:
      - known_samples (dùng train model + build gallery)
      - unknown_samples (dùng tune threshold)
    """
    samples = load_train_meta(train_meta_path, img_root)
    tracks = build_tracks(samples)
    cleaned_samples = clean_tracks(tracks, mode=bbox_mode)

    all_labels = [s.label_id for s in cleaned_samples]
    known_labels, unknown_labels = split_labels_open_set(
        all_labels, unknown_ratio=unknown_ratio
    )

    known_samples = [s for s in cleaned_samples if s.label_id in known_labels]
    unknown_samples = [s for s in cleaned_samples if s.label_id in unknown_labels]

    return known_samples, unknown_samples, known_labels, unknown_labels


def build_val_tracklets_from_unknown(
    unknown_samples: List[BBoxSample],
) -> List[Tracklet]:
    """
    Tạo tracklet cho các label thuộc unknown (val set).
    Ở đây, tracklet = track ground-truth theo (quarter, angle, label_id).
    """
    tracks = build_tracks(unknown_samples)  # key = (quarter, angle, label)
    tracklets: List[Tracklet] = []
    for key, track in tracks.items():
        # convert BBoxSample -> TestBBox để tái dùng hàm get_tracklet_embedding
        for t in [track]:  # mỗi track = 1 tracklet
            tbboxes: List[TestBBox] = []
            for i, s in enumerate(t):
                tbboxes.append(
                    TestBBox(
                        idx=i,  # dummy
                        quarter=s.quarter,
                        angle=s.angle,
                        session=s.session,
                        frame=s.frame,
                        x=s.x,
                        y=s.y,
                        w=s.w,
                        h=s.h,
                        img_path=s.img_path,
                    )
                )
            tr = Tracklet(bboxes=tbboxes, pred_label=-1)
            tracklets.append(tr)
    return tracklets


def tune_threshold(
    pl_module: PlayerReIDModule,
    gallery: Dict[int, torch.Tensor],
    known_samples: List[BBoxSample],
    unknown_tracklets: List[Tracklet],
    image_size: int = 224,
    device: str = "cuda",
    thresholds: List[float] = [0.4, 0.45, 0.5, 0.55, 0.6],
) -> float:
    """
    Simple tuning: maximize F1 giữa {known vs -1} trên một set "giả lập":
      - known_samples: coi như tracklet 1-frame cần predict label đúng
      - unknown_tracklets: phải predict -1
    => Đây là demo, bạn có thể làm phức tạp hơn.
    """
    from collections import defaultdict
    from metric import macro_f1

    device = torch.device(device)
    pl_module.eval().to(device)

    # 1. embedding centroid gallery đã có

    # 2. build "known tracklets" đơn giản: mỗi sample là 1 tracklet
    known_tracklets: List[Tracklet] = []
    for s in known_samples:
        tb = TestBBox(
            idx=0,
            quarter=s.quarter,
            angle=s.angle,
            session=s.session,
            frame=s.frame,
            x=s.x,
            y=s.y,
            w=s.w,
            h=s.h,
            img_path=s.img_path,
        )
        known_tracklets.append(Tracklet(bboxes=[tb], pred_label=s.label_id))

    best_T = thresholds[0]
    best_f1 = -1.0

    for T in thresholds:
        y_true: List[int] = []
        y_pred: List[int] = []

        # known: phải ra 1 trong known_labels
        for tr in known_tracklets[:500]:  # limit cho nhanh
            emb = get_tracklet_embedding(pl_module, tr, image_size=image_size, device=device)
            best_lab = None
            best_sim = -1.0
            for lab, c in gallery.items():
                sim = float(compute_cosine_similarity(emb, c))
                if sim > best_sim:
                    best_sim = sim
                    best_lab = lab
            pred = best_lab if best_sim >= T else -1
            y_true.append(tr.pred_label)  # là label_id thật
            y_pred.append(pred)

        # unknown: phải ra -1
        for tr in unknown_tracklets[:500]:
            emb = get_tracklet_embedding(pl_module, tr, image_size=image_size, device=device)
            best_lab = None
            best_sim = -1.0
            for lab, c in gallery.items():
                sim = float(compute_cosine_similarity(emb, c))
                if sim > best_sim:
                    best_sim = sim
                    best_lab = lab
            pred = best_lab if best_sim >= T else -1
            y_true.append(-1)
            y_pred.append(pred)

        y_true_tensor = torch.tensor(y_true, dtype=torch.long)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.long)
        f1 = float(macro_f1(y_pred_tensor, y_true_tensor))

        if f1 > best_f1:
            best_f1 = f1
            best_T = T

    print(f"[tune_threshold] best T = {best_T}, F1 = {best_f1:.4f}")
    return best_T


def enforce_frame_constraint(
    test_bboxes: List[TestBBox],
    label_preds: List[int],
    pred_scores: List[float],
) -> List[int]:
    """
    Một ID known chỉ gán cho 1 bbox / frame.
    Label -1 thì không giới hạn.
    """
    from collections import defaultdict

    groups: Dict[Tuple[str, str, int, int], List[int]] = defaultdict(list)
    for b in test_bboxes:
        key = (b.quarter, b.angle, b.session, b.frame)
        groups[key].append(b.idx)

    for key, idx_list in groups.items():
        # group theo predicted_label
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in idx_list:
            lab = label_preds[idx]
            label_to_indices[lab].append(idx)

        for lab, idxs in label_to_indices.items():
            if lab == -1:
                continue
            if len(idxs) <= 1:
                continue
            # giữ bbox có score cao nhất
            best_idx = max(idxs, key=lambda i: pred_scores[i])
            for i in idxs:
                if i != best_idx:
                    label_preds[i] = -1

    return label_preds


def run_inference(
    ckpt_path: str,
    train_meta_path: str,
    train_img_root: str,
    test_meta_path: str,
    test_img_root: str,
    bbox_mode: str = "drop",
    unknown_ratio: float = 0.2,
    device: str = "cuda",
    image_size: int = 224,
    submission_path: str = "submission.csv",
):
    """
    Pipeline:
      1. Split known/unknown từ train (open-set).
      2. Build gallery từ known_samples.
      3. Build unknown_tracklets (val) -> tune threshold.
      4. Chạy inference trên test_meta: tracklet -> label/-1.
      5. Enforce frame constraint -> submission.
    """
    # 1. load model
    device = torch.device(device)
    pl_module = PlayerReIDModule.load_from_checkpoint(ckpt_path)
    pl_module.eval().to(device)

    # 2. open-set split
    known_samples, unknown_samples, known_labels, unknown_labels = build_open_set_splits(
        train_meta_path=train_meta_path,
        img_root=train_img_root,
        bbox_mode=bbox_mode,
        unknown_ratio=unknown_ratio,
    )

    # 3. build gallery
    known_samples_side = [s for s in known_samples if s.angle == "side"]
    print("[run_inference] Building gallery...")
    gallery = build_gallery(
        pl_module,
        samples=known_samples_side, # side only
        batch_size=64,
        image_size=image_size,
        device=device,
    )

    # 4. build unknown_tracklets để tune threshold
    print("[run_inference] Building unknown tracklets for threshold tuning...")
    unknown_tracklets = build_val_tracklets_from_unknown(unknown_samples)
    T_unknown = tune_threshold(
        pl_module,
        gallery,
        known_samples=known_samples,
        unknown_tracklets=unknown_tracklets,
        image_size=image_size,
        device=device,
    )

    # 5. load test_meta & build tracklets
    print("[run_inference] Building test tracklets...")
    test_bboxes = load_test_meta(test_meta_path, test_img_root)
    tracklets = build_test_tracklets(test_bboxes)

    # 6. predict label cho mỗi tracklet
    label_preds = [-1] * len(test_bboxes)
    pred_scores = [0.0] * len(test_bboxes)

    print("[run_inference] Predicting labels for tracklets...")
    for tr in tqdm(tracklets):
        emb = get_tracklet_embedding(pl_module, tr, image_size=image_size, device=device)
        best_lab = -1
        best_sim = -1.0
        for lab, c in gallery.items():
            sim = float(compute_cosine_similarity(emb, c))
            if sim > best_sim:
                best_sim = sim
                best_lab = lab
        if best_sim < T_unknown:
            tr.pred_label = -1
        else:
            tr.pred_label = best_lab
        tr.pred_score = best_sim

        # map xuống từng bbox
        for b in tr.bboxes:
            label_preds[b.idx] = tr.pred_label
            pred_scores[b.idx] = tr.pred_score

    # 7. enforce frame constraint
    print("[run_inference] Enforcing frame constraint...")
    label_preds = enforce_frame_constraint(test_bboxes, label_preds, pred_scores)

    # 8. write submission
    print(f"[run_inference] Writing submission to {submission_path}")
    with open(submission_path, "w") as f:
        for lab in label_preds:
            f.write(str(lab) + "\n")
