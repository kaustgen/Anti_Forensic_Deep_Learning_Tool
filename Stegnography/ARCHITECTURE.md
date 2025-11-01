# Hybrid Spatial+Frequency CNN Architecture Documentation

## Overview

This document explains the architecture of the `SpatialFrequencyHybridCNN` model implemented in `sten_dct_cnn.py`.

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [FilterBank (15 HPF Filters)](#filterbank)
3. [TLU Activation](#tlu-activation)
4. [Spatial Branch](#spatial-branch)
5. [DCT Branch](#dct-branch)
6. [Fusion Classifier](#fusion-classifier)
7. [Dataset Classes](#dataset-classes)
8. [Key Design Decisions](#key-design-decisions)

---

## Model Architecture

### High-Level Pipeline

```
Input: RGB Image [256×256] + JPEG DCT Coefficients [32×32×8×8]
           |                            |
           v                            v
    SPATIAL BRANCH                 DCT BRANCH
  (FilterBank + CNNs)          (DC/AC Norm + CNNs)
           |                            |
      256D features                128D features
           |                            |
           +----------------------------+
                        |
                 Concatenate [384D]
                        |
                FUSION CLASSIFIER
                  (FC Layers)
                        |
                   [2D Logits]
              (Clean vs Stego)
```

---

## FilterBank

### Purpose
The FilterBank applies 15 hand-crafted high-pass filters (HPFs) to spatial pixels. These filters were designed by steganalysis researchers to detect subtle artifacts left by steganographic embedding.

### Filter Composition
| Filter Type | Count | Purpose |
|-------------|-------|---------|
| **KV (Square 5×5)** | 1 | Non-directional edge detection |
| **Edge 5×5** | 4 | Directional edges (left, right, top, bottom) |
| **KP (Point Filter)** | 1 | High-pass point filter (identity - low-pass) |
| **KB (Edge Detection)** | 1 | 3×3 edge filter |
| **2D Gabor** | 8 | Oriented texture filters (4 angles × 2 phases) |
| **Total** | **15** | |

### Why These Filters?

1. **Edge Filters** detect inconsistencies in pixel transitions (steganography changes edge sharpness)
2. **Point Filters** highlight noise residuals (embedding modifies noise structure)
3. **Gabor Filters** capture oriented textures (embedding disturbs texture patterns)

### Input/Output
- **Input**: `[B, 3, 256, 256]` - RGB pixels
- **Output**: `[B, 45, 256, 256]` - 15 filter responses per channel (3 × 15 = 45)

### InstanceNorm2d Stabilization
After filtering, **InstanceNorm2d** normalizes outputs to `[-3, +3]`:
- **Before**: Filter outputs can be `±50` (causes gradient explosion)
- **After**: Normalized to `±3` (stable training)

---

## TLU Activation

### Full Name
**Truncated Linear Unit** (TLU)

### Formula
```python
TLU(x) = clamp(x, min=-T, max=T)
```

### Purpose
- Standard ReLU: `max(0, x)` → throws away negative values (loses information)
- TLU: `clamp(x, -3, +3)` → keeps both positive AND negative (preserves noise structure)

### Why Use TLU for Steganalysis?
Steg analysis needs to detect **noise changes**, not just edges:
- Steganography modifies **both positive and negative** noise components
- ReLU would discard half the signal
- TLU preserves full noise spectrum while preventing explosion

### Usage in Model
- **Spatial Branch**: TLU used in Layers 1-2 (after HPF filters)
- **DCT Branch**: ReLU used (frequency domain is already normalized)

---

## Spatial Branch

### Architecture (11 Layers, SRNet-inspired)

#### Type 1: Noise Residual Extraction (Layers 1-2)
```
Input: [B, 45, 256, 256] (FilterBank output)
  ↓
Conv2d(45 → 64) + BN + TLU(T=3.0)     # Layer 1
  ↓
Conv2d(64 → 16) + BN + TLU(T=3.0)     # Layer 2
  ↓
Output: [B, 16, 256, 256]
```
**Why reduce to 16 channels?** Prepares for unpooled segment (saves memory while preserving spatial resolution)

#### Type 2: Unpooled Residual Layers (Layers 3-7)
```
[B, 16, 256, 256]
  ↓
Conv-BN-ReLU + Conv-BN (+ Residual) × 5    # Layers 3-7
  ↓
[B, 16, 256, 256]  (NO POOLING!)
```
**Why no pooling?** Must preserve 256×256 resolution to detect subtle spatial artifacts.

#### Type 3: Pooled Residual Layers (Layers 8-11)
```
[B, 16, 256, 256]
  ↓
Conv-BN-ReLU + Pool (256×256 → 128×128)    # Layer 8
  ↓
Conv-BN-ReLU × 2 (128×128 → 64×64)         # Layers 9-11
  ↓
[B, 256, 64, 64]
```
**Why pool now?** After learning spatial patterns, safe to reduce dimensions for higher-level features.

#### Type 4: Global Pooling (Layer 12)
```
[B, 256, 64, 64]
  ↓
AdaptiveAvgPool2d(1)    # Global Average Pooling
  ↓
[B, 256]  (Spatial feature vector)
```

---

## DCT Branch

### DC/AC Normalization

**Problem**: DCT coefficients have vastly different scales:
- **DC coefficient** (top-left of 8×8 block): Average intensity (~100-200)
- **AC coefficients** (other 63 positions): Detail/texture (~-50 to +50)

**Solution**: Separate z-score normalization:
```python
# DC (position 0,0): Normalize separately
dc_normalized = (dc - dc.mean()) / dc.std()

# AC (positions 0,1 to 7,7): Normalize separately
ac_normalized = (ac - ac.mean()) / ac.std()
```

**Why this matters**: Without normalization, BatchNorm fails (DC dominates variance).

### Architecture (11 Layers, SRNet-inspired)

Same structure as Spatial Branch, but:
- **Input**: `[B, 192, 32, 32]` (3 channels × 64 DCT coeffs = 192)
- **Output**: `[B, 128]` (DCT feature vector)
- **Smaller output** (128D vs 256D) because frequency patterns are more compact

---

## Fusion Classifier

### Architecture
```
[B, 384]  (256 spatial + 128 DCT)
  ↓
FC(384 → 256) + BN + ReLU + Dropout(0.3)
  ↓
FC(256 → 128) + BN + ReLU + Dropout(0.3)
  ↓
FC(128 → 2) + Weight Clipping (±2.0)
  ↓
[B, 2]  (Logits for Clean vs Stego)
```

### Weight Clipping (CRITICAL FIX)
```python
# After each training step:
final_layer.weight.clamp_(-2.0, 2.0)
```

**Why this is critical**:
- Without clipping: Final layer weights → ±5.0 → Logits → ±30 → Loss explosion
- With clipping: Weights stay bounded → Logits stay `[-6, 6]` → Stable training

### Xavier Initialization (gain=0.5)
```python
# Final layer initialization:
nn.init.xavier_uniform_(final_layer.weight, gain=0.5)
```

**Why gain=0.5?**
- Standard Xavier: `gain=1.0` → weights too large for final layer
- Reduced gain: `gain=0.5` → smaller initial weights → prevents early explosion

---

## Dataset Classes

### StegoDCTSpatialDataset
**Returns**: DCT coefficients only
- **Output**: `[C, H, W, 8, 8]` - DCT blocks
- **Use case**: DCT-only models

### HybridSpatialDCTDataset (extends StegoDCTSpatialDataset)
**Returns**: Both spatial pixels AND DCT coefficients
- **Output 1**: `[3, 256, 256]` - RGB pixels (normalized to [0, 1])
- **Output 2**: `[3, 32, 32, 8, 8]` - DCT coefficients (raw)
- **Output 3**: `int` - Label (0=clean, 1=stego)
- **Use case**: Hybrid models

### Data Augmentation
```python
if train:
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        spatial_img = TF.hflip(spatial_img)
        dct_blocks = torch.flip(dct_blocks, dims=[2])  # Flip width dim
    
    # Random vertical flip (50% chance)
    if random.random() > 0.5:
        spatial_img = TF.vflip(spatial_img)
        dct_blocks = torch.flip(dct_blocks, dims=[1])  # Flip height dim
```

**Critical**: Spatial and DCT flips must be **synchronized** (same random seed).

### Caching System
**Problem**: Extracting DCT from JPEG is slow (~50ms per image)

**Solution**: Cache extracted DCT blocks to disk
```python
cache_file = cache_dir / f"{sha1(img_path).hexdigest()}_spatial.pt"
if cache_file.exists():
    dct_blocks = torch.load(cache_file)
else:
    dct_blocks = extract_dct_blocks(img_path)
    torch.save(dct_blocks, cache_file)
```

**Result**: 50× faster training after first epoch.

---

## Key Design Decisions

### 1. Why 256D Spatial vs 128D DCT?
- **Spatial patterns are more complex** (edges, textures, local structures)
- **Frequency patterns are more compact** (DCT already summarizes information)
- **2:1 ratio empirically works best**

### 2. Why InstanceNorm for HPF but BatchNorm for CNNs?
- **InstanceNorm**: Per-sample normalization (good for variable image content)
- **BatchNorm**: Per-batch normalization (good for stable CNN layers)
- **HPF outputs vary wildly** across images → InstanceNorm
- **CNN features are consistent** within batch → BatchNorm

### 3. Why Separate DC/AC Normalization?
- **DC coefficient**: ~100-200 (image brightness)
- **AC coefficients**: ~-50 to +50 (texture detail)
- **Without separation**: BatchNorm uses combined variance (DC dominates)
- **With separation**: Each normalized independently (both contribute equally)

### 4. Why Weight Clipping Instead of Just Gradient Clipping?
- **Gradient clipping**: Limits gradient magnitude (prevents explosion during backprop)
- **Weight clipping**: Limits parameter magnitude (prevents logit explosion during forward pass)
- **Both needed**: Gradient clipping alone doesn't prevent weights from growing over time

### 5. Why 11 Layers in Each Branch?
- **SRNet used 12 layers** (1 input + 11 residual)
- **We use 11 layers** to match SRNet architecture
- **Type distribution**:
  - Type 1 (noise extraction): 2 layers
  - Type 2 (unpooled residual): 5 layers
  - Type 3 (pooled residual): 4 layers
  - Type 4 (global pooling): 1 operation

### 6. Why TF32 Instead of FP16?
- **FP16**: Requires gradient scaling, can have overflow issues
- **TF32**: Automatic on Ampere GPUs, no code changes needed
- **Performance**: Similar speed (~10-20% faster than FP32)
- **Stability**: Better than FP16 (no loss scaling edge cases)

---

## Performance Expectations

### DCT-Only Model
- **Expected Accuracy**: 53-60% on J-UNIWARD (0.1-0.4 bpnzAC)
- **Reason**: Frequency domain alone misses spatial artifacts

### Spatial-Only Model  
- **Expected Accuracy**: 60-67% on J-UNIWARD
- **Reason**: Spatial domain alone misses frequency patterns

### Hybrid Spatial+DCT Model
- **Expected Accuracy**: 68-78% on J-UNIWARD
- **Reason**: Combines both domains → captures complementary information

---

## Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 32 | Dual input (spatial+DCT) uses more VRAM |
| **Learning Rate** | 1e-4 | Conservative for deep hybrid model |
| **Optimizer** | Adam | Better for varying gradient scales |
| **Scheduler** | CosineAnnealing | Smooth LR decay (1e-4 → 1e-6) |
| **Gradient Clipping** | max_norm=1.0 | Prevents gradient explosion |
| **Weight Clipping** | ±2.0 (final layer) | Prevents logit explosion |
| **Dropout** | 0.3 (fusion only) | Prevents overfitting in classifier |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Early Stopping** | Patience=15 | Stop if no improvement |

---

## References

1. **SRNet**: Boroumand, M., Chen, M., & Fridrich, J. (2017). "Deep Residual Network for Steganalysis of Digital Images." *IEEE Transactions on Information Forensics and Security*.

2. **XuNet**: Xu, G., Wu, H. Z., & Shi, Y. Q. (2016). "Structural Design of Convolutional Neural Networks for Steganalysis." *IEEE Signal Processing Letters*.

3. **Thesis**: "Enhancing the performance of convolutional neural network image-based steganalysis in spatial domain using Spatial Rich Model and 2D Gabor filters"
   https://bearworks.missouristate.edu/cgi/viewcontent.cgi?article=4762&context=theses

4. **J-UNIWARD**: Holub, V., Fridrich, J., & Denemark, T. (2014). "Universal Distortion Function for Steganography in an Arbitrary Domain." *EURASIP Journal on Information Security*.

---

## Common Issues and Fixes

### Issue 1: Gradient Explosion
**Symptoms**: Loss > 5.0, logits ±30+, NaN values
**Fix**: 
- ✅ Added gradient clipping (max_norm=1.0)
- ✅ Added weight clipping on final layer (±2.0)
- ✅ Reduced Xavier gain to 0.5 for final layer

### Issue 2: Validation Accuracy Stuck at 50%
**Possible causes**:
- Model predicting same class for everything
- Check class balance in training data
- Increase learning rate slightly (try 2e-4)

### Issue 3: VRAM Overflow
**Solution**: Reduce batch_size from 32 → 24 or 16

### Issue 4: Slow Training
**Solutions**:
- Enable TF32 (already done)
- Increase num_workers to 12 (already done)
- Use DCT cache (already implemented)
- Reduce target_blocks from 32×32 → 24×24

---

## File Structure

```
Stegnography/
├── sten_dct_cnn.py          # Model + Dataset classes
├── train_hybrid.py          # Training script
├── ARCHITECTURE.md          # This file
├── .dct_cache/              # Cached DCT blocks
└── models_hybrid_lowrate/   # Saved model checkpoints
    └── best_hybrid_model.pth
```

---

**Last Updated**: October 29, 2025  
**Author**: Kaleb Austgen
