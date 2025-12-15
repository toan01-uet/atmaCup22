# ✅ Optimization Implementation Checklist

Use this checklist to apply all performance optimizations to your training pipeline.

## 📋 Pre-Training Setup (Run Once)

- [ ] **Generate negative samples CSV**
  ```bash
  python 00_prepare_negatives.py
  ```
  - Creates: `inputs/train_negatives.csv`
  - Time: ~1-2 minutes
  - Only need to run once (reuse for all training runs)

- [ ] **Verify CSV was created**
  ```bash
  ls -lh inputs/train_negatives.csv
  # Should show file size (e.g., 2.5M)
  ```

## 🔧 Code Updates

### Option A: Using CV Training (Recommended)

- [ ] **Update your training script** (e.g., `01_run_cv_train.py`)
  
  Find where you call `build_open_set_splits_cv()` and add these parameters:
  ```python
  from src.inference import build_open_set_splits_cv
  
  known, unknown, known_labels, unknown_labels = build_open_set_splits_cv(
      train_meta_path="inputs/atmaCup22_2nd_meta/train_meta.csv",
      img_root="inputs/images",
      bbox_mode="drop",
      n_folds=5,
      fold_idx=0,
      unknown_per_fold=2,
      cv_seed=42,
      add_bg_negatives=True,
      add_partial_negatives=True,
      # ✅ ADD THESE TWO LINES:
      use_precomputed_negatives=True,
      neg_csv_path="inputs/train_negatives.csv"
  )
  ```

- [ ] **Update DataModule configuration**
  
  Increase `num_workers` for better performance:
  ```python
  from src.data.datamodule import PlayerReIDDataModuleCV
  
  datamodule = PlayerReIDDataModuleCV(
      train_meta_path="inputs/atmaCup22_2nd_meta/train_meta.csv",
      img_root="inputs/images",
      batch_size=64,
      num_workers=8,  # ✅ CHANGE: Increase from 4 to 8 (or more if CPU allows)
      train_ratio=0.8,
      bbox_mode="drop",
      image_size=224,
      n_folds=5,
      fold_idx=0,
      unknown_per_fold=2,
      cv_seed=42,
      use_crop_cache=True,  # ✅ VERIFY: Should be True
      expand_ratio=1.2,
  )
  ```

### Option B: Using Non-CV Training

- [ ] **Update training script** (e.g., `01_run_train.py`)
  ```python
  from src.inference import build_open_set_splits
  
  known, unknown, known_labels, unknown_labels = build_open_set_splits(
      train_meta_path="inputs/atmaCup22_2nd_meta/train_meta.csv",
      img_root="inputs/images",
      bbox_mode="drop",
      unknown_ratio=0.2,
      add_bg_negatives=True,
      add_partial_negatives=True,
      # ✅ ADD THESE TWO LINES:
      use_precomputed_negatives=True,
      neg_csv_path="inputs/train_negatives.csv"
  )
  ```

- [ ] **Update DataModule** (same as Option A above)

## 🧪 Testing & Verification

- [ ] **Test data loading speed**
  ```python
  import time
  
  start = time.time()
  datamodule.setup()
  print(f"Setup time: {time.time() - start:.2f}s")
  # Should be < 60s (was 300-420s before)
  ```

- [ ] **Verify negatives loaded correctly**
  ```python
  print(f"Known samples: {len(known)}")
  print(f"Unknown samples: {len(unknown)}")
  # Unknown should include negatives (label_id=-1)
  ```

- [ ] **Check DataLoader performance**
  ```python
  train_loader = datamodule.train_dataloader()
  
  start = time.time()
  for batch_idx, (images, labels) in enumerate(train_loader):
      if batch_idx == 10:
          break
  elapsed = time.time() - start
  print(f"10 batches: {elapsed:.2f}s ({elapsed/10:.2f}s per batch)")
  ```

## 📊 Expected Results

After applying all optimizations, you should see:

| Metric | Before | After | ✓ |
|--------|--------|-------|---|
| Initial data loading | 5-7 minutes | <1 minute | [ ] |
| First epoch | Baseline | 20-30% faster | [ ] |
| Subsequent epochs | Baseline | 50-70% faster | [ ] |
| Worker restarts | Every epoch | Never (persistent) | [ ] |
| CPU utilization | Spiky | Consistent | [ ] |

## 🚀 Advanced Optimizations (Optional)

### If you have pre-cropped images:

- [ ] **Use CroppedBBoxDataset**
  ```python
  from src.data.dataset import CroppedBBoxDataset
  
  dataset = CroppedBBoxDataset(
      img_paths=your_cropped_paths,
      labels=your_labels,
      transform=transform,
      use_cache=True,
      cache_size=10000
  )
  ```

### If you have lots of RAM:

- [ ] **Enable image cache**
  ```python
  dataset = CroppedBBoxDataset(
      ...,
      use_cache=True,
      cache_size=20000  # Increase cache size
  )
  ```

### If you have powerful CPU:

- [ ] **Increase workers and prefetch**
  ```python
  datamodule = PlayerReIDDataModuleCV(
      ...,
      num_workers=16,  # More workers
  )
  # prefetch_factor is already set to 2 automatically
  ```

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: Negatives CSV not found"
- [ ] Run: `python 00_prepare_negatives.py`
- [ ] Check file exists: `ls inputs/train_negatives.csv`

### Issue: "RuntimeError: Too many open files"
- [ ] Reduce num_workers: `num_workers=4` (instead of 8)
- [ ] Increase system limit: `ulimit -n 4096`

### Issue: Out of memory
- [ ] Reduce batch_size: `batch_size=32` (instead of 64)
- [ ] Reduce cache_size: `cache_size=5000` (instead of 10000)
- [ ] Disable cache: `use_cache=False`

### Issue: Still slow data loading
- [ ] Verify `use_crop_cache=True` in DataModule
- [ ] Check disk I/O (SSD vs HDD)
- [ ] Monitor CPU usage during training (should be high)
- [ ] Try reducing augmentations temporarily to test

## 📈 Performance Monitoring

Add this to your training script to monitor performance:

```python
import time
from collections import defaultdict

times = defaultdict(list)

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()
        # Training step...
        batch_time = time.time() - batch_start
        times['batch'].append(batch_time)
    
    epoch_time = time.time() - epoch_start
    times['epoch'].append(epoch_time)
    
    print(f"Epoch {epoch}: {epoch_time:.2f}s")
    print(f"  Avg batch: {sum(times['batch'][-100:])/100:.3f}s")

print("\n📊 Performance Summary:")
print(f"First epoch: {times['epoch'][0]:.2f}s")
print(f"Avg subsequent: {sum(times['epoch'][1:])/len(times['epoch'][1:]):.2f}s")
print(f"Speedup: {times['epoch'][0]/sum(times['epoch'][1:])*len(times['epoch'][1:]):.1f}x")
```

## ✅ Final Verification

Once everything is working:

- [ ] Training starts in <60 seconds (data loading)
- [ ] No "generating negatives" messages during training
- [ ] Epochs 2+ are significantly faster than epoch 1
- [ ] CPU usage is consistent and high during training
- [ ] GPU utilization is high (not waiting for data)
- [ ] No repeated worker initialization messages

## 🎉 Success!

If all checks pass, you've successfully optimized your training pipeline!

**Time saved per training run:** 10-20 minutes
**Efficiency gain:** 50-70% faster epochs
**Happiness:** 📈😊

---

📖 Need more details? See:
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - Quick overview
- [docs/PERFORMANCE_OPTIMIZATION.md](docs/PERFORMANCE_OPTIMIZATION.md) - Full documentation
- [example_optimized_training.py](example_optimized_training.py) - Code examples
