# atmaCup22 - Basketball Player Re-identification

Unified pipeline for open-set player re-identification with cross-validation.

## 📋 Overview

This project implements a **robust ReID system** with:
- **5-fold cross-validation** with open-set (unknown player) handling
- **Sample-level splitting** (not label-level) for consistent train/val
- **Unified metadata** saved during training, reused in inference
- **Side-angle focused** training with optional multi-view fine-tuning
- **Pre-computed crop cache** and negatives for fast training

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (One-time)
```bash
# Prepare crop cache and negative samples
python prepare_data.py
```
This will:
- Pre-crop and cache all training images from `inputs/atmaCup22_2nd/images/` to `inputs/train_crops_cache/`
- Generate negative samples (background + partial) to `inputs/train_negatives.csv`

**Note:** Test data in `inputs/atmaCup22_2nd/crops/` is already pre-cropped and ready to use.

### 3. Train All Folds
```bash
# Train 5 folds with CV, saves fold metadata automatically
python run_cv_train.py

# Optional: With Stage 2 multi-view fine-tuning
python run_cv_train_with_finetune.py
```

### 4. Run Inference
```bash
python inference_cv.py --fold 0
python inference_cv.py --fold 1
# ... and so on for each fold
```

---

## 📊 Complete Data Pipeline

The data flows through three main stages:

### STAGE 1: PREPARE DATA (`prepare_data.py`)
```
prepare_data.py
├─ Input:  inputs/atmaCup22_2nd/images/        (full training frames)
├─ Read:   inputs/atmaCup22_2nd/train_meta.csv (bbox annotations)
└─ Output: 
    ├─ inputs/train_crops_cache/     (cached training crops)
    └─ inputs/train_negatives.csv    (negative samples)
```

**Purpose:** Pre-process training data to speed up model training
- Crops bounding boxes from full training frames
- Caches crops to disk for ~3-5x faster data loading
- Generates hard negatives (background and partial occlusions)

### STAGE 2: TRAIN (`run_cv_train.py` or `run_cv_train_with_finetune.py`)
```
Training Pipeline
├─ Read:   inputs/atmaCup22_2nd/train_meta.csv (metadata)
├─ Load:   inputs/train_crops_cache/           (cached crops)
├─ Load:   inputs/train_negatives.csv          (negatives)
├─ Process: 5-fold cross-validation
│   ├─ Save: outputs/fold_metadata/fold{i}_train_samples.csv
│   ├─ Save: outputs/fold_metadata/fold{i}_val_samples.csv
│   ├─ Save: outputs/fold_metadata/fold{i}_unknown_labels.csv
│   └─ Save: wandb/ (model checkpoints)
└─ Output: 5 trained models (one per fold)
```

**Purpose:** Train player re-identification models with cross-validation
- **Stage 1:** Main training (side-view, 12 epochs)
- **Stage 2:** Optional multi-view fine-tuning (side+top, 5 epochs)
- Saves fold metadata for consistent inference

### STAGE 3: INFERENCE (`inference_cv.py`)
```
Inference Pipeline
├─ Load:   inputs/atmaCup22_2nd/test_meta.csv  (test annotations)
├─ Load:   inputs/atmaCup22_2nd/crops/         (pre-cropped test images)
├─ Load:   outputs/fold_metadata/              (fold splits from training)
├─ Load:   models/reid-fold{i}.ckpt            (trained model)
├─ Process: For each fold:
│   ├─ Build gallery from training samples
│   ├─ Tune threshold on validation data
│   ├─ Predict labels for test tracklets
│   └─ Apply open-set threshold
└─ Output: outputs/submission_fold{i}.csv      (predictions)
```

**Purpose:** Generate predictions using trained models
- Uses same fold splits as training for consistency
- Applies open-set recognition (unknown player detection)
- Can ensemble predictions from multiple folds

---

## 🏗️ Architecture

### Pipeline Flow

```
DATA PREP → TRAIN (CV) → SAVE FOLD METADATA → INFERENCE (CV) → ENSEMBLE
```

### Key Components

1. **Data Module** (`src/data/datamodule.py`)
   - `PlayerReIDDataModuleCV`: Handles CV fold splitting
   - Splits samples (not labels) for train/val
   - Saves fold metadata to `outputs/fold_metadata/`

2. **Model** (`src/models/pl_module.py`)
   - Backbone: ResNet18/50
   - ArcFace head for metric learning
   - Triplet loss for additional constraint

3. **Inference** (`src/inference.py`)
   - `build_open_set_splits_cv()`: Loads saved fold metadata
   - `tune_threshold()`: Finds optimal threshold per fold
   - Gallery matching with cosine similarity

---

## 📊 Cross-Validation Strategy

### Fold Split Logic

For each fold `i` (out of 5 folds):

1. **Unknown labels**: 2 labels assigned to fold `i` as "open-set" unknowns
2. **Known labels**: All remaining labels
3. **Known samples** split into:
   - `train_known_i`: ~80% of samples per label → train model
   - `val_known_i`: ~20% of samples per label → validation + threshold tuning
4. **Unknown samples**: 
   - Unknown player samples (from unknown labels)
   - Background negatives (from `train_negatives.csv`)
   - Partial negatives (from `train_negatives.csv`)

### Why Sample-Level Splitting?

**Old approach** (label-level):
```
known_labels → split → train_labels, val_labels
Problem: val may not contain all labels
```

**New approach** (sample-level):
```
known_samples → split per label → train_samples, val_samples
Benefit: Both train and val contain all known labels
```

### Fold Metadata Saved

During training, these files are saved to `outputs/fold_metadata/`:
- `fold{i}_train_samples.csv`: Training samples with original label IDs
- `fold{i}_val_samples.csv`: Validation samples with original label IDs
- `fold{i}_unknown_labels.csv`: Unknown label IDs for this fold

**Critical**: Inference loads these files to ensure 100% consistency.

---

## 🔧 Training Details

### Stage 1: Side-Only Training

```python
# From 01_run_cv_train.py
model = PlayerReIDModule(
    num_classes=num_classes,
    backbone_name="resnet18",
    embedding_dim=512,
    arcface_s=30.0,
    arcface_m=0.5,
    lr=1e-4,
    weight_decay=1e-5,
    lambda_triplet=1.0,
)
```

**Loss**: ArcFace + Triplet (lambda=1.0)
**Data**: Side-angle only
**Augmentation**: 
- RandomHorizontalFlip
- ColorJitter
- Resize to 224x224

### Stage 2: Multi-View Fine-tuning (Optional)

```bash
python 01_run_cv_train_with_finetune.py
```

**Additional**: Multi-view consistency loss between side/top pairs
**Benefit**: Better cross-view matching

---

## 🔍 Inference Strategy

### Gallery Building

```python
# For fold i:
# 1. Load fold metadata
train_samples, val_samples, unknown_labels = load_fold_metadata(fold_idx=i)

# 2. Combine train + val as "known" samples
known_samples = train_samples + val_samples

# 3. Build gallery (centroid per known label, side-only)
gallery = build_gallery(model, known_samples_side)
```

### Threshold Tuning

```python
# Use val_known samples as "known" queries
known_tracklets = build_tracklets(val_samples)

# Use unknown player + negatives as "unknown" queries
unknown_tracklets = build_tracklets(unknown_samples)

# Find optimal threshold
threshold = tune_threshold(model, gallery, known_tracklets, unknown_tracklets)
```

### Test Prediction

```python
# For each test tracklet:
# 1. Extract embedding
embedding = get_tracklet_embedding(model, tracklet)

# 2. Match against gallery
similarities = cosine_similarity(embedding, gallery)
best_label, best_score = max(similarities)

# 3. Apply threshold
if best_score >= threshold:
    prediction = best_label
else:
    prediction = -1  # unknown
```

---

## ⚡ Performance Optimizations

### 1. Crop Cache
Pre-crop and cache all training images:
```bash
python 00_prepare_crop_cache.py
```
**Benefit**: ~3-5x faster data loading

### 2. Pre-computed Negatives
Generate negatives once, load from CSV:
```bash
python 00_prepare_negatives.py
```
**Benefit**: 5-7 minutes → <1 second

### 3. DataLoader Settings
```python
num_workers=8
pin_memory=True
persistent_workers=True
prefetch_factor=2
```

---

## 📁 Project Structure

```
atmaCup22/
├── prepare_data.py               # Pre-process crops and generate negatives
├── run_cv_train.py               # Train 5-fold CV
├── run_cv_train_with_finetune.py # Train with Stage 2 multi-view fine-tuning
├── inference_cv.py               # Run inference for each fold
├── inputs/
│   ├── atmaCup22_2nd/
│   │   ├── train_meta.csv        # Training metadata (Q1-Q4 with bbox coords)
│   │   ├── test_meta.csv         # Test metadata with rel_path
│   │   ├── images/               # Full training frame images (for cropping)
│   │   └── crops/                # Test crops and some training crops
│   ├── train_crops_cache/        # Generated by prepare_data.py (cached crops)
│   │   ├── Q1-000/
│   │   │   └── side/
│   │   ├── Q1-001/
│   │   │   └── side/
│   │   └── ... (all quarters: Q1-Q4)
│   └── train_negatives.csv       # Generated by prepare_data.py
├── outputs/
│   ├── fold_metadata/            # Saved during training
│   │   ├── fold0_train_samples.csv
│   │   ├── fold0_val_samples.csv
│   │   ├── fold0_unknown_labels.csv
│   │   └── ... (for each fold 0-4)
│   └── logs/                     # Training logs
├── src/
│   ├── data/
│   │   ├── datamodule.py         # PlayerReIDDataModuleCV
│   │   └── dataset.py
│   ├── models/
│   │   ├── pl_module.py          # PlayerReIDModule
│   │   ├── config.yaml           # Model configuration
│   │   └── model.py
│   ├── utils/
│   │   ├── utils.py              # Data utils + fold metadata I/O
│   │   ├── crop_cache.py         # Crop cache utilities
│   │   ├── loss.py               # Loss functions
│   │   ├── metric.py             # Metrics
│   │   └── logger.py
│   └── inference.py              # Inference utilities
├── notebooks/
│   ├── check_data_label.ipynb
│   └── cropped_images/
├── pyproject.toml
├── requirements.txt
├── README.md
└── REFACTORING_SUMMARY.md
```

## 📂 Input Data Structure - atmaCup22_2nd/

The `atmaCup22_2nd` folder contains both training and test data:

### Training Data: `atmaCup22_2nd/images/`

Full frame images for training (used by `train_meta.csv`):

```
atmaCup22_2nd/images/
├── Q1-000__side__00__01.jpg
├── Q1-000__side__00__02.jpg
├── ... (continues with pattern: {quarter}__{angle}__{session:02d}__{frame:02d}.jpg)
├── Q1-000__side__00__20.jpg
├── Q1-000__side__01__01.jpg
├── ... (all quarters Q1-000 to Q4-019, all sessions, all frames)
└── Q4-019__side__06__30.jpg
```

**Key Points:**
- **Filename format**: `{quarter}__{angle}__{session:02d}__{frame:02d}.jpg`
- **Quarters covered**: Q1-000 to Q4-019 (all training quarters)
- **Angles**: side and top views
- **Sessions**: 0-6 (varies by quarter)
- **Frames**: Variable per session (typically 1-50)
- **Purpose**: Full frame images used for training; bboxes are cropped from these

### Test Data: `atmaCup22_2nd/crops/`

Pre-cropped test images organized by quarter and session:

```
atmaCup22_2nd/crops/
├── Q4-000/
│   ├── sess_0001/
│   │   ├── side/
│   │   │   ├── Q4-000__sess0001__hash1.jpg
│   │   │   ├── Q4-000__sess0001__hash2.jpg
│   │   │   └── ... (hash-based unique names)
│   │   └── top/
│   │       └── ... (if available)
│   ├── sess_0002/
│   └── ... (sessions 1-7)
├── Q4-001/
├── ... (Q4-000 to Q4-019, test quarters only)
└── Q4-019/
    └── ... (sessions and angles)
```

**Key Points:**
- **Quarters covered**: Q4-000 to Q4-019 (test quarters only, Q1-Q3 not in crops)
- **Session format**: `sess_0001`, `sess_0002`, ..., `sess_0007`
- **Angles**: side and top views (not all sessions have both)
- **Filename format**: Hash-based unique identifiers
- **rel_path column**: Test metadata points to these with relative paths like `crops/Q4-000/sess_0001/side/hash.jpg`

### Metadata Files

- **train_meta.csv**: Contains bbox annotations for training frames
  - Columns: `quarter, angle, session, frame, x, y, w, h, label_id`
  - References images in the `images/` folder
  - Covers Q1-Q4 (all quarters)

- **test_meta.csv**: Contains bbox annotations for test crops
  - Columns: `x, y, w, h, quarter, session_no, frame_in_session, angle, rel_path`
  - Direct `rel_path` pointing to cropped test images in `crops/`
  - Covers Q4 only (test set)

---

## 🔬 Implementation Notes

### Key Data Paths

The project uses these default paths:
```
Train metadata:  inputs/atmaCup22_2nd/train_meta.csv
Test metadata:   inputs/atmaCup22_2nd/test_meta.csv
Train images:    inputs/atmaCup22_2nd/images/          # Full frames for training
Test crops:      inputs/atmaCup22_2nd/crops/           # Pre-cropped test images
Cache output:    inputs/train_crops_cache/             # Cached training crops
Negatives CSV:   inputs/train_negatives.csv
Fold metadata:   outputs/fold_metadata/
```

**Important:**
- Training data uses **full frame images** from `images/` folder with bbox coordinates in `train_meta.csv`
- Test data uses **pre-cropped images** from `crops/` folder with direct paths in `test_meta.csv`
- During training, crops are cached to `train_crops_cache/` for faster loading

All paths are relative to the project root directory.

### Key Risks Addressed

1. ✅ **Train/Inference Mismatch**: Unified metadata ensures same splits
2. ✅ **Overfitting Unknown**: 2 unknown labels per fold, diverse negatives
3. ✅ **Label Leakage**: Sample-level split ensures all labels in train/val
4. ✅ **Threshold Instability**: Tuned per fold on consistent val set
5. ✅ **Correct Data Paths**: All scripts use `inputs/atmaCup22_2nd/` for actual data

### Design Decisions

- **Why sample-level split?** Ensures both train and val see all known labels
- **Why save fold metadata?** 100% consistency between train and inference
- **Why side-only?** Test is heavily side-angle biased
- **Why 5 folds?** Balance between unknown diversity and training data size
- **Why single prepare_data.py?** Unified data preparation pipeline for both crops and negatives

---

## 🎯 Next Steps

1. **Create CV-based inference script**: `02_inference_cv.py`
2. **Implement ensemble**: Average predictions across 5 folds
3. **Tune IoU/frame_gap**: Adjust tracklet building parameters
4. **Add bbox jitter**: Augment train with noisy bbox for robustness

---

## 📚 References

- ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
- Triplet Loss: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering"
- Open-set Recognition: Bendale & Boult, "Towards Open Set Deep Networks"

---

## 📝 License

MIT
