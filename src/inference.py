# inference.py

from typing import List, Dict, Tuple
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.utils.utils import (
    BBoxSample,
    TestBBox,
    iou_bbox,
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
    # add neg samples
    generate_background_negatives,
    generate_partial_negatives,
    load_negatives_from_csv,  # NEW: load pre-computed negatives
    # cv label folds
    make_label_folds
)
from src.models.pl_module import PlayerReIDModule

def build_open_set_splits_cv(
    train_meta_path: str,
    img_root: str,
    bbox_mode: str,
    n_folds: int,
    fold_idx: int,
    unknown_per_fold: int = 2,
    cv_seed: int = 42,
    add_bg_negatives: bool = True,
    add_partial_negatives: bool = True,
    neg_csv_path: str = "inputs/train_negatives.csv",  # NEW: path to pre-computed negatives
    use_precomputed_negatives: bool = True,  # NEW: use pre-computed instead of generating
) -> Tuple[List[BBoxSample], List[BBoxSample], List[int], List[int]]:
    """
    Open-set split theo fold:
      - Với fold_idx: chọn đúng unknown_per_fold labels làm unknown.
      - Các label còn lại = known.
      - unknown_samples gồm:
            + sample của unknown_labels
            + optional background negatives
            + optional partial negatives
    
    Performance optimization:
      - Set use_precomputed_negatives=True to load negatives from CSV (FAST)
      - Set use_precomputed_negatives=False to generate on-the-fly (SLOW, 5-7 minutes)
      - Run: python 00_prepare_negatives.py first to generate the CSV
    """
    samples = load_train_meta(train_meta_path, img_root)
    tracks = build_tracks(samples)
    cleaned_samples = clean_tracks(tracks, mode=bbox_mode)

    all_labels = sorted({s.label_id for s in cleaned_samples})
    folds_unknown = make_label_folds(
        unique_labels=all_labels,
        n_folds=n_folds,
        unknown_per_fold=unknown_per_fold,
        seed=cv_seed,
    )

    assert 0 <= fold_idx < n_folds
    unknown_labels = set(folds_unknown[fold_idx])
    known_labels = [lab for lab in all_labels if lab not in unknown_labels]

    # 1. sample của unknown_labels (player chưa thấy trong train fold này)
    unknown_player_samples = [s for s in cleaned_samples if s.label_id in unknown_labels]
    known_samples = [s for s in cleaned_samples if s.label_id in known_labels]

    unknown_samples = list(unknown_player_samples)

    # 2 & 3. Add negatives - use pre-computed or generate on-the-fly
    if use_precomputed_negatives:
        # FAST PATH: Load pre-computed negatives from CSV (~100-500ms)
        try:
            neg_samples = load_negatives_from_csv(neg_csv_path)
            
            # Filter negatives based on flags
            if add_bg_negatives and add_partial_negatives:
                unknown_samples.extend(neg_samples)
            elif add_bg_negatives:
                # Assume background negatives have some distinguishing feature
                # For simplicity, just add all (you can add metadata to CSV to filter)
                unknown_samples.extend(neg_samples)
            elif add_partial_negatives:
                unknown_samples.extend(neg_samples)
                
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Falling back to on-the-fly generation (will be slow)...")
            use_precomputed_negatives = False
    
    if not use_precomputed_negatives:
        # SLOW PATH: Generate negatives on-the-fly (5-7 minutes)
        # 2. thêm background negatives
        if add_bg_negatives:
            bg_neg = generate_background_negatives(
                samples=cleaned_samples,
                num_bg_per_frame=1,
            )
            unknown_samples.extend(bg_neg)

        # 3. thêm partial negatives
        if add_partial_negatives:
            partial_neg = generate_partial_negatives(
                samples=cleaned_samples,
                num_partial_per_box=1,
                max_iou_with_gt=0.4,
            )
            unknown_samples.extend(partial_neg)

    return known_samples, unknown_samples, known_labels, list(unknown_labels)

def build_open_set_splits(
    train_meta_path: str,
    img_root: str,
    bbox_mode: str = "drop",
    unknown_ratio: float = 0.2,
    add_bg_negatives: bool = True,
    add_partial_negatives: bool = True,
    neg_csv_path: str = "inputs/train_negatives.csv",  # NEW: path to pre-computed negatives
    use_precomputed_negatives: bool = True,  # NEW: use pre-computed instead of generating
):
    """
    Build open-set splits for training/validation.
    
    Performance optimization:
      - Set use_precomputed_negatives=True to load negatives from CSV (FAST)
      - Set use_precomputed_negatives=False to generate on-the-fly (SLOW)
      - Run: python 00_prepare_negatives.py first to generate the CSV
    """
    samples = load_train_meta(train_meta_path, img_root)
    tracks = build_tracks(samples)
    cleaned_samples = clean_tracks(tracks, mode=bbox_mode)

    all_labels = [s.label_id for s in cleaned_samples]
    known_labels, unknown_labels = split_labels_open_set(
        all_labels, unknown_ratio=unknown_ratio
    )

    # 1. unknown player (label chưa thấy)
    unknown_player_samples = [s for s in cleaned_samples if s.label_id in unknown_labels]
    known_samples = [s for s in cleaned_samples if s.label_id in known_labels]

    unknown_samples = list(unknown_player_samples)

    # 2 & 3. Add negatives - use pre-computed or generate on-the-fly
    if use_precomputed_negatives:
        # FAST PATH: Load pre-computed negatives from CSV (~100-500ms)
        try:
            neg_samples = load_negatives_from_csv(neg_csv_path)
            
            # Filter negatives based on flags
            if add_bg_negatives and add_partial_negatives:
                unknown_samples.extend(neg_samples)
            elif add_bg_negatives:
                unknown_samples.extend(neg_samples)
            elif add_partial_negatives:
                unknown_samples.extend(neg_samples)
                
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Falling back to on-the-fly generation (will be slow)...")
            use_precomputed_negatives = False
    
    if not use_precomputed_negatives:
        # SLOW PATH: Generate negatives on-the-fly
        # 2. thêm background negatives (không có cầu thủ)
        if add_bg_negatives:
            bg_negatives = generate_background_negatives(
                samples=cleaned_samples,
                num_bg_per_frame=1,
            )
            unknown_samples.extend(bg_negatives)

        # 3. thêm partial negatives (chỉ 1 phần cơ thể)
        if add_partial_negatives:
            partial_negatives = generate_partial_negatives(
                samples=cleaned_samples,
                num_partial_per_box=1,
                max_iou_with_gt=0.4,
            )
            unknown_samples.extend(partial_negatives)

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
    from src.utils.metric import macro_f1

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


from src.utils.utils import TestBBox, iou_bbox  # nhớ import iou_bbox từ utils.py


def enforce_frame_constraint(
    test_bboxes: List[TestBBox],
    label_preds: List[int],
    pred_scores: List[float],
    iou_same_player: float = 0.5,
) -> List[int]:
    """
    Rule mới:
      - Cho phép N bbox cùng label trong 1 frame nếu chúng thuộc cùng "cluster overlap" (IoU >= iou_same_player).
      - Nếu cùng label nhưng nằm ở nhiều cluster rời rạc -> chỉ giữ cluster có score cao nhất, các cluster khác gán -1.
      - Label -1 không bị giới hạn.
    """
    # group indices theo (quarter, angle, session, frame)
    frame_groups: Dict[Tuple[str, str, int, int], List[int]] = defaultdict(list)
    for b in test_bboxes:
        key = (b.quarter, b.angle, b.session, b.frame)
        frame_groups[key].append(b.idx)

    # để access nhanh TestBBox theo idx
    idx_to_bbox: Dict[int, TestBBox] = {b.idx: b for b in test_bboxes}

    for key, idx_list in frame_groups.items():
        # group theo predicted label
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in idx_list:
            lab = label_preds[idx]
            label_to_indices[lab].append(idx)

        for lab, idxs in label_to_indices.items():
            if lab == -1:
                # unknown không bị ràng buộc
                continue
            if len(idxs) <= 1:
                # chỉ 1 bbox mang label này trong frame -> OK
                continue

            # --- Cluster các bbox theo overlap IoU >= iou_same_player ---
            # build adjacency graph
            n = len(idxs)
            visited = [False] * n
            clusters: List[List[int]] = []

            for i in range(n):
                if visited[i]:
                    continue
                # BFS/DFS nhỏ để lấy component
                stack = [i]
                visited[i] = True
                comp = [idxs[i]]
                while stack:
                    u = stack.pop()
                    bu = idx_to_bbox[idxs[u]]
                    for v in range(n):
                        if visited[v]:
                            continue
                        bv = idx_to_bbox[idxs[v]]
                        if iou_bbox(bu, bv) >= iou_same_player:
                            visited[v] = True
                            stack.append(v)
                            comp.append(idxs[v])
                clusters.append(comp)

            if len(clusters) <= 1:
                # chỉ 1 cluster -> tất cả bbox overlap nhau -> giữ hết
                continue

            # --- chọn cluster có score cao nhất, các cluster khác set -1 ---
            cluster_scores: List[float] = []
            for comp in clusters:
                max_score = max(pred_scores[i] for i in comp)
                cluster_scores.append(max_score)

            # index cluster giữ lại
            best_cluster_idx = max(range(len(clusters)), key=lambda ci: cluster_scores[ci])
            keep_indices = set(clusters[best_cluster_idx])

            for ci, comp in enumerate(clusters):
                if ci == best_cluster_idx:
                    continue
                # drop các bbox trong cluster yếu hơn
                for idx in comp:
                    label_preds[idx] = -1

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
    logger = None,
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
    if logger:
        logger.info("\nLoading model from checkpoint...")
        logger.info(f"Checkpoint: {ckpt_path}")
    device = torch.device(device)
    pl_module = PlayerReIDModule.load_from_checkpoint(ckpt_path)
    pl_module.eval().to(device)
    if logger:
        logger.info(f"Model loaded successfully on device: {device}")

    # 2. open-set split
    if logger:
        logger.info("\nBuilding open-set splits...")
        logger.info(f"Unknown ratio: {unknown_ratio}")
    known_samples, unknown_samples, known_labels, unknown_labels = build_open_set_splits(
        train_meta_path=train_meta_path,
        img_root=train_img_root,
        bbox_mode=bbox_mode,
        unknown_ratio=unknown_ratio,
    )
    if logger:
        logger.info(f"Known samples: {len(known_samples)}, Known labels: {len(known_labels)}")
        logger.info(f"Unknown samples: {len(unknown_samples)}, Unknown labels: {len(unknown_labels)}")

    # 3. build gallery
    known_samples_side = [s for s in known_samples if s.angle == "side"]
    if logger:
        logger.info(f"\nBuilding gallery from {len(known_samples_side)} side-angle samples...")
    print("[run_inference] Building gallery...")
    gallery = build_gallery(
        pl_module,
        samples=known_samples_side, # side only
        batch_size=64,
        image_size=image_size,
        device=device,
    )
    if logger:
        logger.info(f"Gallery built with {len(gallery)} classes")

    # 4. build unknown_tracklets để tune threshold
    if logger:
        logger.info("\nBuilding unknown tracklets for threshold tuning...")
    print("[run_inference] Building unknown tracklets for threshold tuning...")
    unknown_tracklets = build_val_tracklets_from_unknown(unknown_samples)
    if logger:
        logger.info(f"Built {len(unknown_tracklets)} unknown tracklets")
        logger.info("Tuning threshold...")
    T_unknown = tune_threshold(
        pl_module,
        gallery,
        known_samples=known_samples,
        unknown_tracklets=unknown_tracklets,
        image_size=image_size,
        device=device,
    )
    if logger:
        logger.info(f"Best threshold: {T_unknown:.4f}")

    # 5. load test_meta & build tracklets
    if logger:
        logger.info("\nLoading test metadata and building tracklets...")
    print("[run_inference] Building test tracklets...")
    test_bboxes = load_test_meta(test_meta_path, test_img_root)
    if logger:
        logger.info(f"Loaded {len(test_bboxes)} test bounding boxes")
    tracklets = build_test_tracklets(test_bboxes)
    if logger:
        logger.info(f"Built {len(tracklets)} test tracklets")

    # 6. predict label cho mỗi tracklet
    label_preds = [-1] * len(test_bboxes)
    pred_scores = [0.0] * len(test_bboxes)

    if logger:
        logger.info("\nPredicting labels for tracklets...")
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
    if logger:
        logger.info("\nEnforcing frame constraint...")
    print("[run_inference] Enforcing frame constraint...")
    label_preds = enforce_frame_constraint(test_bboxes, label_preds, pred_scores)
    
    # Count predictions
    from collections import Counter
    pred_counts = Counter(label_preds)
    if logger:
        logger.info(f"Prediction distribution:")
        logger.info(f"  - Unknown (-1): {pred_counts.get(-1, 0)} ({pred_counts.get(-1, 0)/len(label_preds)*100:.2f}%)")
        logger.info(f"  - Known labels: {len(label_preds) - pred_counts.get(-1, 0)} ({(len(label_preds) - pred_counts.get(-1, 0))/len(label_preds)*100:.2f}%)")
        logger.info(f"  - Unique known labels predicted: {len([k for k in pred_counts.keys() if k != -1])}")

    # 8. write submission
    if logger:
        logger.info(f"\nWriting submission to {submission_path}")
    print(f"[run_inference] Writing submission to {submission_path}")
    with open(submission_path, "w") as f:
        for lab in label_preds:
            f.write(str(lab) + "\n")
    if logger:
        logger.info(f"Submission file created successfully with {len(label_preds)} predictions")

def run_inference_cv(
    ckpt_path: str,
    train_meta_path: str,
    train_img_root: str,
    test_meta_path: str,
    test_img_root: str,
    bbox_mode: str = "drop",
    n_folds: int = 5,
    fold_idx: int = 0,
    unknown_per_fold: int = 2,
    cv_seed: int = 42,
    device: str = "cuda",
    image_size: int = 224,
    submission_path: str = "submission_fold0.csv",
):
    """
    Inference cho 1 fold cụ thể:
      - fold_idx xác định 2 label nào là unknown.
      - Training model fold_idx phải được train bằng PlayerReIDDataModuleCV với cùng fold.
    """
    device = torch.device(device)
    pl_module = PlayerReIDModule.load_from_checkpoint(ckpt_path)
    pl_module.eval().to(device)

    # 1. open-set split theo fold
    known_samples, unknown_samples, known_labels, unknown_labels = build_open_set_splits_cv(
        train_meta_path=train_meta_path,
        img_root=train_img_root,
        bbox_mode=bbox_mode,
        n_folds=n_folds,
        fold_idx=fold_idx,
        unknown_per_fold=unknown_per_fold,
        cv_seed=cv_seed,
        add_bg_negatives=True,
        add_partial_negatives=True,
    )

    # 2. gallery từ known_samples (có thể filter angle=="side" nếu muốn)
    known_samples_side = [s for s in known_samples if s.angle == "side"]
    gallery = build_gallery(
        pl_module,
        samples=known_samples_side,
        batch_size=64,
        image_size=image_size,
        device=device,
    )

    # 3. unknown_tracklets cho threshold tuning
    unknown_tracklets = build_val_tracklets_from_unknown(unknown_samples)

    T_unknown = tune_threshold(
        pl_module,
        gallery,
        known_samples=known_samples_side,
        unknown_tracklets=unknown_tracklets,
        image_size=image_size,
        device=device,
    )

    # 4. test tracklets
    test_bboxes = load_test_meta(test_meta_path, test_img_root)
    tracklets = build_test_tracklets(test_bboxes)

    # 5. predict tracklet labels
    label_preds = [-1] * len(test_bboxes)
    pred_scores = [0.0] * len(test_bboxes)

    for tr in tracklets:
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

        for b in tr.bboxes:
            label_preds[b.idx] = tr.pred_label
            pred_scores[b.idx] = tr.pred_score

    # 6. enforce frame constraint mới (cho phép overlap cùng ID)
    label_preds = enforce_frame_constraint(test_bboxes, label_preds, pred_scores)

    # 7. viết submission
    with open(submission_path, "w") as f:
        for lab in label_preds:
            f.write(str(lab) + "\n")
