# atmaCup22 - Basketball Player Re-identification

Open-set player re-identification for basketball games with multi-view camera angles (side + top views).

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Pipeline Overview](#-pipeline-overview)
- [Data Processing Flow](#-data-processing-flow)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Inference Pipeline](#-inference-pipeline)
- [Performance Optimization](#-performance-optimization)
- [Multi-GPU Training & Inference](#️-multi-gpu-training--inference)

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (One-time Setup)
```bash
# Generate crop cache for faster training
python 00_prepare_crop_cache.py

# Generate negative samples (background + partial crops)
python 00_prepare_negatives.py
```

### 3. Train Model
```bash
# Single GPU training (default)
python 01_run_cv_train.py

# With Stage 2 multi-view fine-tuning (recommended for mixed-angle data)
python 01_run_cv_train_with_finetune.py

# Multi-GPU training (DataParallel strategy)
GPU_DEVICES="0,1,2,3" python 01_run_cv_train.py

# Or single split training
python 01_run_train.py
```

### 4. Run Inference
```bash
# Single GPU inference
python 02_inference_run.py

# Multi-GPU inference (parallel batch processing)
GPU_DEVICES="0,1,2,3" python 02_inference_run.py
```

---

## 📊 Pipeline Overview

```

                     DATA PREPARATION                             │

  train_meta.csv  →  [Load & Clean]  →  Tracks                   │
       ↓                                    ↓                     │
  Crop Cache      →  [Pre-process]  →  Negative Samples          │

                             ↓

                     MODEL TRAINING                               │

  [Stage 1: Side-only]       │
       ↓                                     ↓                    │
  Feature Extraction  →  [ArcFace + Triplet Loss]                │
       ↓                                     ↓                    │
  Multi-view Pairs  →  [Stage 2: Fine-tune with Consistency]     │

                             ↓

                     INFERENCE                                    │

  Test Tracklets  →  [Extract Embeddings]  →  Gallery Matching   │
       ↓                                          ↓               │
  Cosine Similarity  →  [Threshold]  →  Known/Unknown Decision   │
       ↓                                          ↓               │
  Post-processing  →  [Frame Constraints]  →  Predictions Final  

```

---

## 🔄 Data Processing Flow

### Step 1: Load & Clean Data
**Module:** `src/utils/utils.py`

```python
# 1. Load train metadata
samples = load_train_meta(train_meta_path, img_root)
# → List[BBoxSample]: quarter, angle, session, frame, bbox, label_id

# 2. Build tracks (group by player/angle/quarter)
tracks = build_tracks(samples)
# → Dict[Tuple[quarter, angle, label_id], List[BBoxSample]]

# 3. Clean bad bounding boxes
cleaned_samples = clean_tracks(tracks, mode="drop")
# mode="drop": Remove outlier boxes
# mode="interp": Interpolate bad boxes
```

### Step 2: Create Crop Cache (Optional but Recommended)
**Script:** `00_prepare_crop_cache.py`

Pre-crops all bounding boxes to avoid runtime cropping:
```python
from src.utils.crop_cache import prepare_cropped_cache

cached_samples = prepare_cropped_cache(
    samples=samples,
    cache_dir="inputs/train_crops_cache",
    expand_ratio=1.2,
    quality=95
)
# Saves cropped images: train_crops_cache/Q1-000/sess_0001/frame_00_crop_0.jpg
```

**Benefits:**
- 5-10x faster data loading during training
- No runtime image cropping overhead
- Reusable across multiple training runs

### Step 3: Generate Negative Samples (Open-set)
**Script:** `00_prepare_negatives.py`

Creates negative samples for open-set recognition:
```python
# Background negatives: Random crops with no players
bg_negatives = generate_background_negatives(samples, num_bg_per_frame=1)

# Partial negatives: Crops with incomplete player bodies
partial_negatives = generate_partial_negatives(
    samples, 
    num_partial_per_box=1,
    max_iou_with_gt=0.4
)
```

Output: `inputs/train_negatives.csv` with label_id=-1

### Step 4: Data Splitting

#### For Cross-Validation (Recommended):
```python
from src.inference import build_open_set_splits_cv

known_samples, unknown_samples, known_labels, unknown_labels = \
    build_open_set_splits_cv(
        train_meta_path=train_meta_path,
        img_root=img_root,
        bbox_mode="drop",
        n_folds=5,
        fold_idx=0,  # 0-4 for 5-fold CV
        unknown_per_fold=2,  # 2 labels per fold as unknown
        use_precomputed_negatives=True  # Fast loading
    )
```

#### Data Split Strategy:
- **Known Classes:** Train normal classification (with ArcFace)
- **Unknown Classes:** Validation for open-set threshold tuning
- **Negative Samples:** Train model to reject non-players

---

## 🧠 Model Architecture

**Module:** `src/models/model.py`, `src/models/pl_module.py`

### Backbone
```python
ResNet50 (pretrained on ImageNet)
  ↓
Global Average Pooling
  ↓
Embedding Layer (512-dim)
  ↓
L2 Normalization
```

### Training Heads

1. **ArcFace Head** (Main classification)
   - Margin-based softmax for discriminative features
   - Only for known classes

2. **Triplet Loss** (Metric learning)
   - Pulls same-player embeddings closer
   - Pushes different-player embeddings apart

3. **Consistency Loss** (Multi-view, optional)
   - Enforces similar embeddings for same player from different angles
   - Side-view vs Top-view alignment

### Model Configuration
```python
PlayerReIDModule(
    backbone="resnet50",
    embedding_dim=512,
    num_classes=num_known_labels,  # Dynamic based on known labels
    arcface_s=30.0,  # Scale parameter
    arcface_m=0.5,   # Angular margin
    triplet_margin=0.3,
    lr=0.001,
    weight_decay=5e-4
)
```

---

## 🎯 Training Strategy

### Stage 1: Side-View Only Training (Main Stage)
**Focus:** Train on dominant camera angle (side view)

```python
# DataModule setup
datamodule = PlayerReIDDataModuleCV(
    train_meta_path="inputs/atmaCup22_2nd_meta/train_meta.csv",
    img_root="inputs/images",
    batch_size=64,
    num_workers=8,
    image_size=224,
    n_folds=5,
    fold_idx=0,
    use_crop_cache=True  # Use pre-cropped images
)

# Training - Single GPU
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=1
)
trainer.fit(model, datamodule)

# Training - Multi-GPU (Data Parallel)
# Automatically scales batch size across GPUs
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=[0, 1, 2, 3],  # Use GPUs 0, 1, 2, 3
    strategy="ddp",  # DistributedDataParallel for best performance
    # strategy="dp",  # Alternative: DataParallel (simpler but slower)
)
trainer.fit(model, datamodule)

# Training - Auto-detect all available GPUs
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=-1,  # Use all available GPUs
    strategy="ddp"
)
trainer.fit(model, datamodule)
```

**Loss Function:**
```python
total_loss = arcface_loss + triplet_loss
```

**Hyperparameters:**
- Epochs: 15-30
- Learning Rate: 1e-3 → 1e-4 (with scheduler)
- Batch Size: 64
- Image Size: 224×224

### Stage 2: Multi-View Fine-tuning (Optional)
**Focus:** Add top-view consistency

```python
# Use multi-view pairs
pairs = build_multiview_pairs(samples)
# → [(side_sample, top_sample)] for same player

# Fine-tune with consistency loss
total_loss = (
    1.0 * side_arcface_loss +
    0.5 * top_arcface_loss +
    0.1 * consistency_loss
)
```

**Training Script with Fine-tuning:**
```bash
# Train with automatic Stage 2 fine-tuning after each fold
python 01_run_cv_train.py
```

Or use the dedicated script:
```bash
# More control over fine-tuning parameters
python 01_run_cv_train_with_finetune.py
```

**Fine-tuning Configuration:**
```python
# In 01_run_cv_train_with_finetune.py
ENABLE_FINETUNE = True  # Set to False to skip Stage 2
FINETUNE_EPOCHS = 5
FINETUNE_LR = 3e-5  # Much lower than Stage 1
FINETUNE_CONSISTENCY_WEIGHT = 0.1
```

**How it works:**
1. After Stage 1 training completes for a fold, the best checkpoint is loaded
2. Multi-view pairs are built from training data (side+top views of same player)
3. Model is fine-tuned with lower learning rate (3e-5 vs 1e-4)
4. Consistency loss encourages similar embeddings across views
5. Separate checkpoints saved: `reid-fold{i}-finetuned-epoch-{val_f1}.ckpt`

**Benefits:**
- Improves cross-angle matching (side ↔ top)
- Better generalization to mixed-angle test data
- Minimal overhead (5 epochs, ~5-10 minutes per fold)

**Hyperparameters:**
- Epochs: 3-5 (Stage 1), 3-5 (Stage 2)
- Learning Rate: 1e-4 (Stage 1) → 3e-5 (Stage 2) (very small)
- Loss Weights: Side=1.0, Top=0.5, Consistency=0.1

**Why Stage 2 is Optional:**
- Test data is heavily biased toward side-view
- Multi-view helps but isn't critical
- Stage 1 alone achieves good performance

---

## 🔍 Inference Pipeline

**Script:** `02_inference_run.py`
**Module:** `src/inference.py`

### Step 1: Build Gallery (Known Player Embeddings)
```python
# Extract embeddings for all known players
gallery = build_gallery(
    model=trained_model,
    samples=known_samples,
    transform=transform,
    device="cuda"
)
# → Dict[label_id, embedding_tensor]
```

### Step 2: Process Test Tracklets
```python
# Load test data
test_bboxes = load_test_meta(test_meta_path, img_root)

# Build tracklets (temporal sequences)
tracklets = build_test_tracklets(
    test_bboxes,
    iou_threshold=0.3,
    max_frame_gap=30
)
# → List[Tracklet]: continuous player detections
```

### Step 3: Extract Test Embeddings
```python
for tracklet in tracklets:
    # Get embedding for tracklet (mean of frame embeddings)
    embedding = get_tracklet_embedding(
        model=model,
        tracklet=tracklet,
        transform=transform,
        device="cuda"
    )
```

### Step 4: Gallery Matching
```python
# Compute similarity with all known players
similarities = compute_cosine_similarity(
    query_embedding, 
    gallery_embeddings
)

# Find best match
best_match_id = similarities.argmax()
best_score = similarities[best_match_id]

# Apply threshold for open-set
if best_score >= threshold:
    prediction = best_match_id  # Known player
else:
    prediction = -1  # Unknown player
```

### Step 5: Threshold Tuning
```python
# Tune threshold on validation set
best_threshold = tune_threshold(
    model=model,
    gallery=gallery,
    known_samples=val_known,
    unknown_tracklets=val_unknown,
    thresholds=[0.4, 0.45, 0.5, 0.55, 0.6]
)
# → Select threshold that maximizes F1 score
```

### Inference Output Format
```python
# For each test tracklet
{
    "quarter": "Q4-000",
    "angle": "side",
    "session": 1,
    "start_frame": 5,
    "end_frame": 15,
    "pred_label": 42,  # or -1 for unknown
    "confidence": 0.87
}
```

---

## ⚡ Performance Optimization

**NEW (Dec 2025):** Significant speed improvements implemented!

### Quick Setup (3 Steps)

1. **Generate negatives once:**
   ```bash
   python 00_prepare_negatives.py
   ```

2. **Use pre-computed negatives in training:**
   ```python
   known, unknown, _, _ = build_open_set_splits_cv(
       ...,
       use_precomputed_negatives=True,
       neg_csv_path="inputs/train_negatives.csv"
   )
   ```

3. **Increase DataLoader workers:**
   ```python
   datamodule = PlayerReIDDataModuleCV(..., num_workers=8)
   ```

### Performance Gains

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Data loading | 5-7 min | <1 min | **6-7x faster** |
| First epoch | Baseline | -20~30% | **1.3x faster** |
| Later epochs | Baseline | -50~70% | **2-3x faster** |
| **Total saved** | - | - | **10-20 min/run** |

### Key Optimizations

1. **Pre-computed Negatives** → Avoid 5-7 min generation every run
2. **Persistent Workers** → No worker restart between epochs
3. **Prefetch Factor** → Better GPU utilization
4. **Crop Cache** → No runtime image cropping
5. **LRU Cache** → Cache frequently accessed images


---

## 📁 Project Structure

```
atmaCup22/
 00_prepare_crop_cache.py      # Pre-crop bounding boxes
 00_prepare_negatives.py       # Generate negative samples
 01_run_cv_train.py            # Cross-validation training
 01_run_train.py               # Single split training
 02_inference_run.py           # Run inference on test set
 main.py                       # Legacy main script

 src/
   ├── data/
   │   ├── dataset.py            # Dataset classes (PlayerReIDDataset, CroppedBBoxDataset)
   │   └── datamodule.py         # PyTorch Lightning DataModules
   ├── models/
   │   ├── model.py              # Model architecture (ResNet + ArcFace)
   │   └── pl_module.py          # Lightning module (training logic)
   ├── utils/
   │   ├── utils.py              # Data processing utilities
   │   ├── crop_cache.py         # Crop caching utilities
   │   ├── metric.py             # Evaluation metrics
   │   └── loss.py               # Loss functions
   └── inference.py              # Inference pipeline

 inputs/
   ├── atmaCup22_2nd_meta/       # Metadata CSV files
   ├── images/                   # Raw images
   ├── crops/                    # Test crops
   ├── train_crops_cache/        # Pre-cropped training images
 train_negatives.csv       # Pre-computed negative samples   └─

 docs/
   ├── PERFORMANCE_OPTIMIZATION.md  # Detailed optimization guide
   └── CROP_CACHE.md                # Crop cache documentation

 OPTIMIZATION_SUMMARY.md       # Quick optimization guide
 OPTIMIZATION_CHECKLIST.md     # Step-by-step checklist
 example_optimized_training.py # Code examples
 README.md                     # This file
```

---

## 🔧 Key Configuration

### Training Configuration
- **Backbone:** ResNet50 (ImageNet pretrained)
- **Embedding Dim:** 512
- **ArcFace Margin:** 0.5
- **Triplet Margin:** 0.3
- **Batch Size:** 64 (per GPU, scales with number of GPUs)
- **Learning Rate:** 1e-3 → 1e-4
- **Epochs:** 15-30 (Stage 1), 3-5 (Stage 2)
- **Multi-GPU:** DDP (DistributedDataParallel) or DP (DataParallel)

### Data Configuration
- **Image Size:** 224×224
- **Crop Expand Ratio:** 1.2
- **Train/Val Split:** 80/20
- **CV Folds:** 5
- **Unknown per Fold:** 2 labels

### Inference Configuration
- **Tracklet IoU Threshold:** 0.3
- **Max Frame Gap:** 30
- **Similarity Threshold:** 0.45-0.55 (tuned on validation)
- **Multi-GPU Inference:** Parallel batch processing across GPUs

---

## 🖥️ Multi-GPU Training & Inference

### Training with Multiple GPUs

#### Option 1: Using PyTorch Lightning Trainer (Recommended)

**DDP (DistributedDataParallel) - Best Performance:**
```python
import pytorch_lightning as pl

trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=[0, 1, 2, 3],  # Specify GPU IDs
    strategy="ddp",  # Each GPU gets its own process
    sync_batchnorm=True,  # Sync batch norm across GPUs
    precision=16,  # Mixed precision for faster training (optional)
)

# Effective batch size = batch_size * num_gpus
# E.g., batch_size=64 on 4 GPUs = effective batch_size of 256
trainer.fit(model, datamodule)
```

**DP (DataParallel) - Simpler but Slower:**
```python
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=[0, 1, 2, 3],
    strategy="dp",  # Single process, broadcasts to GPUs
)
trainer.fit(model, datamodule)
```

**Auto-detect All GPUs:**
```python
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=-1,  # Use all available GPUs
    strategy="ddp"
)
trainer.fit(model, datamodule)
```

#### Option 2: Environment Variables
```bash
# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3
python 01_run_cv_train.py

# Or inline
CUDA_VISIBLE_DEVICES=0,1,2,3 python 01_run_cv_train.py
```

### Multi-GPU Training Best Practices

1. **Batch Size Scaling:**
   - Effective batch size = `batch_size × num_gpus`
   - With 4 GPUs and batch_size=64: effective_batch_size=256
   - May need to adjust learning rate: `lr × sqrt(num_gpus)`

2. **Learning Rate Adjustment:**
   ```python
   # Linear scaling rule
   base_lr = 0.001
   num_gpus = 4
   scaled_lr = base_lr * num_gpus  # 0.004
   
   # Or square root scaling (more stable)
   scaled_lr = base_lr * (num_gpus ** 0.5)  # 0.002
   ```

3. **Gradient Accumulation (if limited GPU memory):**
   ```python
   trainer = pl.Trainer(
       max_epochs=20,
       accelerator="gpu",
       devices=[0, 1],
       strategy="ddp",
       accumulate_grad_batches=2,  # Accumulate gradients over 2 batches
   )
   # Effective batch size = batch_size × num_gpus × accumulate_grad_batches
   ```

4. **Mixed Precision Training:**
   ```python
   trainer = pl.Trainer(
       max_epochs=20,
       accelerator="gpu",
       devices=[0, 1, 2, 3],
       strategy="ddp",
       precision=16,  # or "bf16" for bfloat16
   )
   # ~2x faster training, ~50% less memory
   ```

### Inference with Multiple GPUs

#### Option 1: Parallel Batch Processing
```python
import torch
from torch.nn.parallel import DataParallel

# Wrap model with DataParallel
model = PlayerReIDModule.load_from_checkpoint(checkpoint_path)
model = DataParallel(model, device_ids=[0, 1, 2, 3])
model.eval()

# Process batches in parallel
for batch in test_loader:
    with torch.no_grad():
        embeddings = model(batch)  # Automatically distributed across GPUs
```

#### Option 2: Manual GPU Distribution
```python
import torch.multiprocessing as mp

def inference_worker(gpu_id, tracklets_subset, model_path, output_queue):
    """Worker function for each GPU"""
    device = torch.device(f"cuda:{gpu_id}")
    model = PlayerReIDModule.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    
    results = []
    for tracklet in tracklets_subset:
        embedding = get_tracklet_embedding(model, tracklet, device=device)
        results.append((tracklet, embedding))
    
    output_queue.put(results)

# Distribute tracklets across GPUs
num_gpus = 4
tracklets_per_gpu = len(tracklets) // num_gpus

mp.set_start_method('spawn', force=True)
output_queue = mp.Queue()

processes = []
for gpu_id in range(num_gpus):
    start_idx = gpu_id * tracklets_per_gpu
    end_idx = start_idx + tracklets_per_gpu if gpu_id < num_gpus - 1 else len(tracklets)
    tracklets_subset = tracklets[start_idx:end_idx]
    
    p = mp.Process(
        target=inference_worker,
        args=(gpu_id, tracklets_subset, model_path, output_queue)
    )
    p.start()
    processes.append(p)

# Collect results
all_results = []
for _ in range(num_gpus):
    all_results.extend(output_queue.get())

for p in processes:
    p.join()
```

#### Option 3: Using Lightning Predictor
```python
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0, 1, 2, 3],  # Use multiple GPUs
    strategy="ddp"
)

# Lightning automatically distributes inference
predictions = trainer.predict(model, test_dataloader)
```

### Performance Comparison

| Configuration | Training Speed | Memory Usage | Complexity |
|--------------|----------------|--------------|------------|
| Single GPU | 1x (baseline) | 100% | Low |
| 2 GPUs (DDP) | ~1.8x | 50% per GPU | Medium |
| 4 GPUs (DDP) | ~3.5x | 25% per GPU | Medium |
| 4 GPUs (DP) | ~2.8x | Unbalanced | Low |
| 4 GPUs + Mixed Precision | ~6.5x | 12.5% per GPU | Medium |

### Troubleshooting Multi-GPU Issues

**Issue: "CUDA out of memory"**
```python
# Solution 1: Reduce batch size per GPU
datamodule = PlayerReIDDataModuleCV(
    batch_size=32,  # Reduced from 64
    ...
)

# Solution 2: Use gradient accumulation
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Effective batch size = 32 * 4 = 128
    ...
)

# Solution 3: Use mixed precision
trainer = pl.Trainer(
    precision=16,
    ...
)
```

**Issue: "Inconsistent results across GPUs"**
```python
# Solution: Ensure sync_batchnorm=True
trainer = pl.Trainer(
    strategy="ddp",
    sync_batchnorm=True,  # Synchronize batch norm statistics
    ...
)
```

**Issue: "Slow multi-GPU training"**
- Use DDP instead of DP
- Check GPU utilization: `nvidia-smi`
- Increase `num_workers` in DataLoader
- Enable persistent workers and prefetch

**Issue: "Different results between single and multi-GPU"**
```python
# Set deterministic behavior
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)
trainer = pl.Trainer(
    deterministic=True,  # Ensure reproducibility
    ...
)
```

### Example: Complete Multi-GPU Training Script

```python
import pytorch_lightning as pl
from src.data.datamodule import PlayerReIDDataModuleCV
from src.models.pl_module import PlayerReIDModule

# Setup
pl.seed_everything(42, workers=True)

# DataModule
datamodule = PlayerReIDDataModuleCV(
    train_meta_path="inputs/atmaCup22_2nd_meta/train_meta.csv",
    img_root="inputs/images",
    batch_size=64,  # Per GPU batch size
    num_workers=8,  # Per GPU workers
    use_crop_cache=True,
    n_folds=5,
    fold_idx=0
)

# Model
model = PlayerReIDModule(
    backbone="resnet50",
    embedding_dim=512,
    num_classes=datamodule.num_classes,
    lr=0.001 * (4 ** 0.5),  # Scale LR for 4 GPUs
)

# Trainer - Multi-GPU
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=[0, 1, 2, 3],  # 4 GPUs
    strategy="ddp",
    sync_batchnorm=True,
    precision=16,  # Mixed precision
    accumulate_grad_batches=1,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor="val_f1_macro",
            mode="max",
            save_top_k=3
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
)

# Train
trainer.fit(model, datamodule)
```

---

## 📊 Evaluation Metrics

- **Macro F1 Score:** Primary metric for open-set recognition
- **Accuracy:** For known class classification
- **Threshold Sensitivity:** Validation curve for threshold selection

---

## 🎓 References

- ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- Triplet Loss for Person Re-identification
- Open-set Recognition with Threshold Tuning

---

## 📝 Notes

### Training Tips
1. Start with Stage 1 (side-only) until convergence
2. Use cross-validation for robust model selection
3. Monitor validation F1 score for open-set performance
4. Tune threshold on held-out validation set
5. **Multi-GPU:** Use DDP strategy for best performance, scale learning rate with num_gpus

### Data Tips
1. Always run `00_prepare_crop_cache.py` before training
2. Generate negatives once with `00_prepare_negatives.py`
3. Use `use_crop_cache=True` in DataModule for speed
4. **Multi-GPU:** Increase `num_workers` proportionally (e.g., 8 workers per GPU)

### Inference Tips
1. Use side-view gallery for best results
2. Apply frame constraints for temporal consistency
3. Start with threshold=0.5, adjust based on validation
4. Multi-view can be added as post-processing refinement
5. **Multi-GPU:** Distribute tracklets across GPUs for parallel processing

### Multi-GPU Tips
1. **Training:** Use `strategy="ddp"` for best performance (faster than `"dp"`)
2. **Batch Size:** Effective batch size = `batch_size × num_gpus`
3. **Learning Rate:** Scale LR by `sqrt(num_gpus)` or linearly
4. **Memory:** Use mixed precision (`precision=16`) to reduce memory usage
5. **Debugging:** Start with 1 GPU, then scale to multiple GPUs
6. **Monitoring:** Use `nvidia-smi` to check GPU utilization
7. **Inference:** Distribute workload manually or use DataParallel for simplicity
