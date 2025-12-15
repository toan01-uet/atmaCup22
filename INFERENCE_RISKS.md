# Inference Pipeline Risk Analysis

## Current Issue
**ALL 9,223 predictions are -1 (unknown)**

## Critical Risks Leading to All -1 Predictions

### 1. **Threshold Too High** ⚠️ HIGH RISK
- **Location**: `tune_threshold()` function
- **Issue**: If the tuned threshold is too high (e.g., > 0.6), all similarities will fall below it
- **Root Cause**: 
  - Gallery embeddings may not match test embeddings distribution
  - Side-angle only gallery vs mixed-angle test data
  - Model trained on different data distribution
- **Check**: Print `T_unknown` value - if > 0.6, this is likely the issue

### 2. **Empty or Invalid Gallery** ⚠️ HIGH RISK
- **Location**: `build_gallery()` function  
- **Issue**: Gallery has 0 classes or invalid embeddings
- **Root Causes**:
  - No side-angle samples in known_samples
  - All side samples filtered out by bbox_mode="drop"
  - Gallery embeddings are zeros or NaN
- **Check**: Print `len(gallery)` - should be > 0

### 3. **Test Image Path Mismatch** ⚠️ HIGH RISK
- **Location**: `load_test_meta()` in `02_inference_run.py`
- **Issue**: Test data expects `crops/` directory but images don't exist or paths are wrong
- **Current Code**: 
  ```python
  test_img_root = os.path.join(script_dir, "inputs")  # Points to "inputs/"
  ```
- **Problem**: If test images are in wrong location, embeddings will be incorrect
- **Check**: Verify test images exist at expected paths

### 4. **Invalid Bounding Boxes in Test Data** ⚠️ MEDIUM RISK
- **Location**: `get_tracklet_embedding()` 
- **Issue**: All test bboxes are invalid (x2 <= x1 or y2 <= y1), causing empty embeddings
- **Result**: Returns `torch.zeros()` for all tracklets → low similarity → all -1
- **Already Fixed**: Added skip for invalid bboxes, but if ALL are invalid, returns zeros
- **Check**: Print how many bboxes are skipped

### 5. **Model-Data Mismatch** ⚠️ HIGH RISK
- **Location**: Model checkpoint loading
- **Issue**: Model trained on different data/setup than inference
- **Root Causes**:
  - Checkpoint trained with different image preprocessing
  - Checkpoint from wrong fold/experiment
  - Model collapsed during training (all embeddings similar)
- **Check**: Verify checkpoint path and training logs

### 6. **Gallery-Test Angle Mismatch** ⚠️ HIGH RISK
- **Location**: Gallery uses only side-angle, test has mixed angles
- **Issue**: 
  ```python
  known_samples_side = [s for s in known_samples if s.angle == "side"]
  ```
- **Problem**: If test data has many non-side angles, similarity will be low
- **Impact**: Cross-angle matching may fail completely
- **Check**: Print angle distribution in test data

### 7. **Known/Unknown Split Issue** ⚠️ MEDIUM RISK
- **Location**: `build_open_set_splits()`
- **Issue**: All labels marked as unknown, none as known
- **Root Causes**:
  - `unknown_ratio=0.2` but random split puts all test labels in unknown
  - Pre-computed negatives not loaded, fallback generation failed
- **Check**: Print `len(known_labels)` and `len(unknown_labels)`

### 8. **Frame Constraint Over-Filtering** ⚠️ LOW RISK
- **Location**: `enforce_frame_constraint()`
- **Issue**: Constraint logic sets valid predictions back to -1
- **Less Likely**: This runs after prediction, so if all predictions are already -1, this won't make it worse

### 9. **Embedding Extraction Failure** ⚠️ MEDIUM RISK
- **Location**: `get_tracklet_embedding()`
- **Issue**: 
  - All tracklets have empty bboxes list
  - All images fail to load
  - Transform fails silently
- **Result**: Returns zero embeddings → no matches → all -1
- **Check**: Add logging inside `get_tracklet_embedding()`

### 10. **Device/Memory Issues** ⚠️ LOW RISK
- **Location**: Model inference
- **Issue**: GPU OOM causes fallback to CPU or errors produce zero embeddings
- **Less Likely**: Would see errors/crashes
- **Check**: Monitor GPU memory usage

## Diagnostic Script

```python
# Add to 02_inference_run.py after line 448 (after threshold tuning):

print(f"\n{'='*80}")
print("DIAGNOSTIC CHECKS")
print(f"{'='*80}")
print(f"1. Gallery size: {len(gallery)} classes")
print(f"2. Best threshold: {T_unknown:.4f}")
print(f"3. Known labels: {len(known_labels)}")
print(f"4. Unknown labels: {len(unknown_labels)}")
print(f"5. Test tracklets: {len(tracklets)}")
print(f"6. Sample gallery embeddings norm: {[gallery[k].norm().item() for k in list(gallery.keys())[:3]]}")

# Test a single tracklet manually
if len(tracklets) > 0:
    test_tr = tracklets[0]
    test_emb = get_tracklet_embedding(pl_module, test_tr, image_size=image_size, device=device)
    print(f"7. Sample test embedding norm: {test_emb.norm().item():.4f}")
    
    # Compute similarities
    sims = []
    for lab, c in gallery.items():
        sim = float(compute_cosine_similarity(test_emb, c))
        sims.append((lab, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    print(f"8. Top 5 similarities: {sims[:5]}")
    print(f"9. Prediction: {'Unknown' if sims[0][1] < T_unknown else f'Label {sims[0][0]}'}")

print(f"{'='*80}\n")
```

## Recommended Fixes

### Priority 1: Add Diagnostics
Add the diagnostic script above to understand which risk is active.

### Priority 2: Check Test Image Paths
Verify test images are loading correctly:
```python
# In load_test_meta or get_tracklet_embedding
print(f"Loading image: {img_path}")
if not os.path.exists(img_path):
    print(f"ERROR: Image not found: {img_path}")
```

### Priority 3: Lower Threshold Range
Change threshold search range to detect if threshold is the issue:
```python
thresholds: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6]  # Added lower values
```

### Priority 4: Add Multi-Angle Gallery
Instead of side-only, use all angles:
```python
# In run_inference()
# known_samples_side = [s for s in known_samples if s.angle == "side"]  # OLD
gallery = build_gallery(
    pl_module,
    samples=known_samples,  # Use ALL angles, not just side
    batch_size=64,
    image_size=image_size,
    device=device,
)
```

### Priority 5: Validate Checkpoint
Verify the checkpoint is valid and trained properly:
```bash
# Check checkpoint metrics
python -c "
import torch
ckpt = torch.load('atmacup22-reid/3t8w0o4d/checkpoints/reid-fold0-epoch=5-val_f1_macro=0.000.ckpt')
print('Checkpoint keys:', ckpt.keys())
print('Val F1:', ckpt.get('callbacks', {}).get('ModelCheckpoint', {}).get('best_model_score'))
"
```

## Most Likely Culprits (in order):

1. **Threshold too high** (80% probability) - Check if T_unknown > 0.5
2. **Gallery-Test angle mismatch** (70% probability) - Side-only gallery vs mixed test
3. **Test image path issues** (60% probability) - Images not found at expected paths
4. **Model checkpoint issue** (40% probability) - val_f1_macro=0.000 is suspicious
5. **Empty/small gallery** (30% probability) - Not enough side-angle samples

## Quick Fix Priority Order:
1. Add diagnostic prints
2. Use multi-angle gallery (not just side)
3. Lower threshold range
4. Verify test image paths
5. Check checkpoint quality
