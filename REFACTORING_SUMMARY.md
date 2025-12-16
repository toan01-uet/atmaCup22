# Refactoring Summary - atmaCup22

## Changes Made

### 1. **Unified Train/Val/Inference Pipeline**

#### Problem
- Training used label-level splitting: `known_labels → train_labels, val_labels`
- Inference re-split known/unknown with different logic
- No guarantee of consistency between train and inference
- Potential label leakage and mismatch

#### Solution
- **Sample-level splitting**: Split samples per label into train/val (not labels themselves)
- **Saved fold metadata**: During training, save `train_samples`, `val_samples`, `unknown_labels` to disk
- **Inference loads metadata**: Use saved files to ensure 100% consistency

### 2. **Code Changes**

#### `src/utils/utils.py`
**Added:**
- `split_samples_by_ratio_per_label()`: Split samples while keeping all labels in both train/val
- `save_fold_metadata()`: Save fold splits to CSV files
- `load_fold_metadata()`: Load fold splits from CSV files

#### `src/data/datamodule.py`
**Changed:**
- `PlayerReIDDataModuleCV.setup()`:
  - Now splits `known_samples` by samples (not labels)
  - Saves fold metadata to `outputs/fold_metadata/`
  - Added `self.fold_meta_dir` attribute

**Before:**
```python
train_labels, val_labels = split_labels_by_ratio(known_labels, train_ratio=0.8)
train_samples = [s for s in cleaned if s.label_id in train_labels]
val_samples = [s for s in cleaned if s.label_id in val_labels]
```

**After:**
```python
known_samples = [s for s in cleaned if s.label_id in known_labels]
train_samples, val_samples = split_samples_by_ratio_per_label(
    known_samples, train_ratio=0.8, seed=cv_seed
)
save_fold_metadata(fold_idx, train_samples, val_samples, unknown_labels, fold_meta_dir)
```

#### `src/inference.py`
**Changed:**
- `build_open_set_splits_cv()`:
  - Now loads saved fold metadata instead of re-splitting
  - Simplified parameters: only needs `fold_idx` and `fold_meta_dir`
  - Combines `train_samples + val_samples` as "known" for gallery building

**Removed:**
- `build_open_set_splits()`: Old non-CV approach (unused)

**Before:**
```python
def build_open_set_splits_cv(
    train_meta_path, img_root, bbox_mode, n_folds, fold_idx, 
    unknown_per_fold, cv_seed, add_bg_negatives, add_partial_negatives, ...
):
    # Re-load and re-split data
    samples = load_train_meta(...)
    tracks = build_tracks(samples)
    cleaned = clean_tracks(tracks)
    # ... split logic here ...
```

**After:**
```python
def build_open_set_splits_cv(
    fold_idx, fold_meta_dir="outputs/fold_metadata", 
    neg_csv_path="inputs/train_negatives.csv", add_negatives=True
):
    # Load saved metadata
    train_samples, val_samples, unknown_labels = load_fold_metadata(fold_idx, fold_meta_dir)
    known_samples = train_samples + val_samples
    # ... rest of logic ...
```

### 3. **Files Removed**

Deleted unused/redundant files:
- `main.py` (placeholder)
- `example_multiview_finetuning.py` (example)
- `example_optimized_training.py` (example)
- `01_run_train.py` (non-CV training, deprecated)
- `02_inference_run.py` (non-CV inference, deprecated)
- `README.md.backup` (backup)
- `INFERENCE_RISKS.md` (outdated)
- `OPTIMIZATION_CHECKLIST.md` (outdated)
- `OPTIMIZATION_SUMMARY.md` (outdated)
- `docs/` directory (outdated documentation)

### 4. **New Documentation**

Created clean, focused `README.md` with:
- Clear explanation of CV strategy
- Sample-level vs label-level splitting
- Fold metadata workflow
- Pipeline architecture
- Quick start guide
- Implementation notes

### 5. **Remaining Files**

**Scripts:**
- `00_prepare_crop_cache.py` - Pre-process crops (unchanged)
- `00_prepare_negatives.py` - Generate negatives (unchanged)
- `01_run_cv_train.py` - Train 5-fold CV (updated to save metadata)
- `01_run_cv_train_with_finetune.py` - Train with Stage 2 (updated to save metadata)

**Source Code:**
- `src/data/datamodule.py` - Refactored for sample-level splits
- `src/data/dataset.py` - Unchanged
- `src/models/pl_module.py` - Unchanged
- `src/models/model.py` - Unchanged
- `src/utils/utils.py` - Added metadata I/O functions
- `src/utils/crop_cache.py` - Unchanged
- `src/utils/metric.py` - Unchanged
- `src/utils/loss.py` - Unchanged
- `src/inference.py` - Refactored to load metadata

### 6. **Benefits**

✅ **Consistency**: Train and inference use identical splits
✅ **No re-splitting**: Inference loads saved metadata, no logic duplication
✅ **Better validation**: Both train and val contain all known labels
✅ **Cleaner code**: Removed ~1000 lines of redundant docs/examples
✅ **Clear pipeline**: Single source of truth for fold splits

### 7. **TODO**

- [ ] Create `02_inference_cv.py` to use new `build_open_set_splits_cv()`
- [ ] Implement ensemble across 5 folds
- [ ] Update frame constraint to allow multiple overlapping bbox with same ID
- [ ] Add bbox jitter augmentation for robustness

---

## Migration Guide

### For Training
No changes needed! The training scripts automatically save fold metadata.

### For Inference
**Old code:**
```python
from src.inference import build_open_set_splits_cv

known, unknown, known_labels, unknown_labels = build_open_set_splits_cv(
    train_meta_path=...,
    img_root=...,
    bbox_mode="drop",
    n_folds=5,
    fold_idx=0,
    unknown_per_fold=2,
    cv_seed=42,
)
```

**New code:**
```python
from src.inference import build_open_set_splits_cv

known, unknown, known_labels, unknown_labels = build_open_set_splits_cv(
    fold_idx=0,
    fold_meta_dir="outputs/fold_metadata",
    neg_csv_path="inputs/train_negatives.csv",
    add_negatives=True
)
```

---

## Verification

Run these checks to verify the refactoring:

```bash
# 1. Check no syntax errors
python -m py_compile src/data/datamodule.py
python -m py_compile src/inference.py
python -m py_compile src/utils/utils.py

# 2. Test training (should save fold metadata)
python 01_run_cv_train.py  # or run 1 fold only

# 3. Check fold metadata files exist
ls -la outputs/fold_metadata/

# Expected files:
# fold0_train_samples.csv
# fold0_val_samples.csv
# fold0_unknown_labels.csv
# ... (fold1-4)

# 4. Test inference loads metadata (TODO: create inference script)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
├─────────────────────────────────────────────────────────────┤
│  1. DataModule setup()                                      │
│     ├─ Load & clean data                                    │
│     ├─ Define unknown labels (2 per fold)                   │
│     ├─ Filter known_samples                                 │
│     ├─ Split samples: train_known, val_known                │
│     └─ SAVE fold metadata to disk ★                         │
│                                                              │
│  2. Train model on train_known                              │
│  3. Validate on val_known                                   │
│  4. Save checkpoint                                          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                           │
├─────────────────────────────────────────────────────────────┤
│  1. Load checkpoint                                          │
│  2. build_open_set_splits_cv()                              │
│     └─ LOAD fold metadata from disk ★                       │
│  3. Build gallery (train_known + val_known, side-only)      │
│  4. Tune threshold                                           │
│     ├─ Known queries: val_known tracklets                   │
│     └─ Unknown queries: unknown player + negatives          │
│  5. Predict test                                             │
│  6. Ensemble across folds                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Insight

**The Problem**: Train and inference were like two separate worlds, each making their own decisions about what's "known" and "unknown".

**The Solution**: Train saves its decisions to disk, inference reads and follows them. Simple, consistent, no surprises.

---

## Final Status

### ✅ Completed Tasks

1. ✅ **Merged prepare scripts**: `00_prepare_data.py` combines crop caching and negative generation
2. ✅ **Created CV inference**: `02_inference_cv.py` uses unified fold metadata
3. ✅ **Removed old files**: Cleaned up deprecated scripts and documentation
4. ✅ **Updated datamodule**: Sample-level splits with metadata persistence
5. ✅ **Refactored inference**: Loads saved metadata for consistency

### 📁 Current File Structure

```
Root Scripts:
├── 00_prepare_data.py              # Unified data preparation
├── 01_run_cv_train.py              # CV training (5 folds)
├── 01_run_cv_train_with_finetune.py # Optional multi-view fine-tuning
└── 02_inference_cv.py              # CV inference using fold metadata

Source Code:
└── src/
    ├── data/
    │   ├── datamodule.py           # Updated with sample-level splits
    │   └── dataset.py
    ├── models/
    │   ├── model.py
    │   └── pl_module.py
    ├── utils/
    │   ├── crop_cache.py
    │   ├── logger.py
    │   ├── utils.py                # Added fold metadata functions
    │   └── ...
    └── inference.py                # Updated CV splits loader
```

### 🚀 Quick Start

```bash
# 1. Prepare all data
python 00_prepare_data.py

# 2. Train all folds
python 01_run_cv_train.py

# 3. Run inference for each fold
for fold in {0..4}; do
    python 02_inference_cv.py --fold $fold
done

# 4. Ensemble results (implement separately)
# Combine outputs/submission_fold*.csv
```

### 🎯 Key Benefits

- **Consistency**: Training and inference use identical splits
- **Reproducibility**: Fold metadata saved to disk
- **Simplicity**: 3 main scripts instead of 7+
- **Clarity**: Removed confusing examples and outdated docs
- **Maintainability**: Single source of truth for CV logic

---

**Date**: December 16, 2025
**Status**: Refactoring Complete ✅
