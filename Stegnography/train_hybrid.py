#!/usr/bin/env python3
"""
Training script for Hybrid Spatial+Frequency CNN.

This combines:
    - Spatial domain: FilterBank HPF on pixels (captures pixel artifacts)
    - Frequency domain: DCT processing (captures frequency patterns)

Expected performance:
    - DCT-only: 53-60% on J-UNIWARD
    - Spatial+DCT Hybrid: 68-78% on J-UNIWARD (big improvement!)

Architecture Overview:
    1. Spatial Branch: RGB pixels → HPF filters → InstanceNorm → 11 SRNet layers → 256D features
    2. DCT Branch: JPEG DCT coeffs → DC/AC normalization → 11 SRNet layers → 128D features
    3. Fusion: Concatenate 256D + 128D → FC layers → Binary classification

Key Optimizations Applied:
    - TF32 enabled for RTX 3080 (FP32 precision, FP16 speed)
    - Gradient clipping (max_norm=1.0) to prevent explosion
    - Weight clipping on final layer (±2.0) to prevent logit explosion
    - CosineAnnealingLR scheduler for smooth convergence
    - Early stopping with patience=15
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import wandb

from sten_dct_cnn import SpatialFrequencyHybridCNN, HybridSpatialDCTDataset

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# GPU OPTIMIZATION: TensorFloat-32 (TF32)
# ============================================================
# Enable TF32 for RTX 3080 (Ampere architecture)
# TF32 provides:
#   - FP32 precision (no accuracy loss)
#   - FP16 speed (uses Tensor Cores)
#   - Automatic operation (no code changes needed)
# 
# Expected speedup: 10-20% faster training compared to pure FP32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logger.info("✓ TensorFloat-32 (TF32) enabled for Ampere GPU")

# ============================================================
# CONFIGURATION
# ============================================================

# Dataset selection
# Toggle between high-rate (0.6-0.8 bpnzAC) and low-rate (0.1-0.4 bpnzAC) datasets
# High-rate: Easier to learn (more signal), faster convergence
# Low-rate: Realistic target (harder), better for final evaluation
USE_HIGH_RATE_DATA = False  # Start with low-rate for realistic evaluation

if USE_HIGH_RATE_DATA:
    # High-rate diagnostic dataset (stronger steganographic signal)
    EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'BOSS_high_rate_diagnostic.xlsx'
    IMG_ROOT = Path(__file__).parent.parent / 'dataGen'
    CACHE_DIR = Path(__file__).parent / '.dct_diagnostic_cache'
    MODEL_DIR = Path(__file__).parent / 'models_hybrid_highrate'
    TEST_NAME = "High-Rate Diagnostic (0.6-0.8 bpnzAC)"
else:
    # Low-rate realistic dataset (subtle steganographic changes)
    #EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'BOSS_stego_training.xlsx'
    EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'BOSS_stego_metadata.xlsx'
    IMG_ROOT = Path(__file__).parent.parent / 'dataGen'
    CACHE_DIR = Path(__file__).parent / '.dct_cache'
    MODEL_DIR = Path(__file__).parent / 'models_hybrid_lowrate'
    #TEST_NAME = "Low-Rate Test (0.1-0.4 bpnzAC)"
    TEST_NAME = "Low-Rate Test (0.1-0.4 bpnzAC) w/ RGB"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

# If you want to start the training from an existing model, use this
RESUME_FROM_CHECKPOINT = True

# Batch size: 32 for RTX 3080 (16 GB VRAM)
# - Smaller than typical (64-128) due to dual input processing (spatial + DCT)
# - Could increase to 64 if only 8-12 GB VRAM used during training
BATCH_SIZE = 32

# Learning rate: 1e-4 (conservative for deep hybrid model)
# - Lower than typical (1e-3) because:
#   1. Model is very deep (11+11 layers = 22 total)
#   2. Hybrid architecture needs careful convergence
#   3. Prevents early explosion in fusion classifier
LEARNING_RATE = 1e-4

# Weight decay: 1e-4 (L2 regularization)
# - Prevents overfitting on small dataset (20K images)
# - Applied to all trainable parameters via Adam optimizer
WEIGHT_DECAY = 1e-4

# Number of epochs: 100 (will early stop if no improvement)
# - Typical steganalysis models need 50-100 epochs to converge
# - Early stopping (patience=15) will halt training if val_acc plateaus
NUM_EPOCHS = 100

# Dropout: 0.3 (applied in fusion classifier only)
# - NOT applied in spatial/DCT branches (would hurt feature learning)
# - Only used in FC layers (256→128→2) to prevent overfitting
DROPOUT = 0.3

# ============================================================
# DATA SPLITS
# ============================================================

# 70/15/15 split (standard for medium-sized datasets)
# - Train: 14,000 images (used for gradient updates)
# - Val: 3,000 images (used for hyperparameter tuning, early stopping)
# - Test: 3,000 images (NEVER seen during training, final evaluation)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# EARLY STOPPING CONFIGURATION
# ============================================================

# Patience: 15 epochs without validation improvement
# - Prevents wasting time if model has converged
# - Saves best model checkpoint during training
# - Typical values: 10-20 epochs for steganalysis
PATIENCE = 15

# ============================================================
# GPU INFORMATION LOGGING
# ============================================================

# Print GPU specifications if available
# Useful for debugging and performance monitoring
if torch.cuda.is_available():
    logger.info("\n" + "="*60)
    logger.info("GPU INFORMATION")
    logger.info("="*60)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    logger.info("="*60 + "\n")

# Create model directory for saving checkpoints
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# TRAINING INITIALIZATION LOG
# ============================================================

logger.info("="*60)
logger.info("HYBRID SPATIAL+FREQUENCY CNN TRAINING")
logger.info("="*60)
logger.info(f"Dataset: {TEST_NAME}")
logger.info(f"Device: {DEVICE}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Learning Rate: {LEARNING_RATE}")
logger.info(f"Epochs: {NUM_EPOCHS}")
logger.info(f"Early Stop Patience: {PATIENCE}")
logger.info("="*60)


# ============================================================
# LOAD DATASET
# ============================================================

logger.info("\nLoading dataset...")

# HybridSpatialDCTDataset returns BOTH:
#   1. Spatial RGB pixels: [3, 256, 256] normalized to [0, 1]
#   2. DCT coefficients: [3, 32, 32, 8, 8] raw JPEG DCT blocks
#
# Key features:
#   - Caches extracted DCT blocks to disk (speeds up training)
#   - Applies random flips during training (spatial + DCT synchronized)
#   - Handles grayscale images (pads Cb/Cr channels with zeros)
dataset = HybridSpatialDCTDataset(
    excel_path=EXCEL_PATH,
    img_root=IMG_ROOT,
    dct_channels=['Y', 'Cb', 'Cr'],  # Use all YCbCr channels
    target_blocks=(32, 32),  # 256×256 pixels = 32×32 blocks of 8×8
    cache_dir=CACHE_DIR,
    train=True  # Enable data augmentation (random flips)
)

logger.info(f"Total samples: {len(dataset)}")

# ============================================================
# SPLIT DATASET
# ============================================================

# Split into train/val/test with fixed random seed for reproducibility
# - random_split ensures no data leakage between splits
# - Generator seed=42 ensures same splits across runs
total_size = len(dataset)
train_size = int(TRAIN_RATIO * total_size)
val_size = int(VAL_RATIO * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible splits
)

logger.info(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")

# ============================================================
# CREATE STRATIFIED WEIGHTED SAMPLER FOR CLASS BALANCE
# ============================================================

# WeightedRandomSampler with stratification ensures:
#   1. 50% clean, 50% stego (class balance)
#   2. Within stego: 25% each of 0.1, 0.2, 0.3, 0.4 bpnzAC (payload balance)
#   3. Within BOTH clean AND stego: maintain RGB/grayscale ratio (source balance)
#
# Example with 10% RGB, 90% grayscale overall:
#   - Clean batch: 10% RGB clean, 90% grayscale clean
#   - Stego batch: 10% RGB stego (2.5% each 0.1/0.2/0.3/0.4), 90% grayscale stego (22.5% each 0.1/0.2/0.3/0.4)

logger.info("\nComputing stratified sample weights...")

# Extract metadata from training split
train_indices = train_dataset.indices
train_metadata = []

for idx in train_indices:
    record = dataset.records.iloc[idx]
    
    # Get label (clean=0, stego=1)
    is_stego = record[dataset.label_col]
    label = 1 if is_stego else 0
    
    # Get RGB flag (column J)
    # RGB column should always exist: True for color, False for grayscale
    is_rgb = bool(record['RGB'])  # Ensure it's a boolean
    
    # Get payload category (0.1, 0.2, 0.3, 0.4) - only for stego images
    if is_stego:
        payload = record['Payload Category']
    else:
        payload = None  # Clean images have no payload
    
    train_metadata.append({
        'label': label,
        'is_rgb': is_rgb,
        'payload': payload
    })

# Convert to tensors for easier manipulation
labels = torch.tensor([m['label'] for m in train_metadata])
is_rgb = torch.tensor([m['is_rgb'] for m in train_metadata])
payloads = [m['payload'] for m in train_metadata]

# ========== COMPUTE STRATIFICATION GROUPS ==========

# Create stratification groups:
#   - Clean RGB
#   - Clean Grayscale
#   - Stego RGB 0.1
#   - Stego RGB 0.2
#   - Stego RGB 0.3
#   - Stego RGB 0.4
#   - Stego Grayscale 0.1
#   - Stego Grayscale 0.2
#   - Stego Grayscale 0.3
#   - Stego Grayscale 0.4

group_indices = {}
for i, meta in enumerate(train_metadata):
    if meta['label'] == 0:  # Clean
        group_key = f"clean_{'rgb' if meta['is_rgb'] else 'gray'}"
    else:  # Stego
        payload_str = str(meta['payload']).replace('.', 'p')
        group_key = f"stego_{'rgb' if meta['is_rgb'] else 'gray'}_{payload_str}"
    
    if group_key not in group_indices:
        group_indices[group_key] = []
    group_indices[group_key].append(i)

# Log group statistics
logger.info("\nDataset stratification groups:")
logger.info("-" * 60)
for group_key, indices in sorted(group_indices.items()):
    logger.info(f"  {group_key:30s}: {len(indices):5d} samples ({100*len(indices)/len(train_metadata):5.2f}%)")
logger.info("-" * 60)

# ========== COMPUTE SAMPLE WEIGHTS ==========

# Goal: Each group should contribute equally to training
# Weight = 1 / (group_size × num_groups)
# This ensures all groups are sampled proportionally

sample_weights = torch.zeros(len(train_metadata))

for group_key, indices in group_indices.items():
    group_size = len(indices)
    # Inverse frequency weighting: smaller groups get higher weights
    group_weight = 1.0 / group_size if group_size > 0 else 0.0
    
    for idx in indices:
        sample_weights[idx] = group_weight

# Normalize weights so they sum to number of samples
# This ensures we sample exactly len(train_metadata) samples per epoch
sample_weights = sample_weights / sample_weights.sum() * len(train_metadata)

logger.info(f"\nSample weights computed:")
logger.info(f"  Min weight: {sample_weights.min():.4f}")
logger.info(f"  Max weight: {sample_weights.max():.4f}")
logger.info(f"  Mean weight: {sample_weights.mean():.4f}")

# Create sampler (samples with replacement to balance all groups)
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True  # Allow re-sampling to balance groups
)

# ============================================================
# CREATE DATALOADERS
# ============================================================

logger.info("Creating dataloaders...")

# Training dataloader
# - Uses weighted sampler (no shuffle since sampler handles randomness)
# - num_workers=12: Parallel data loading (utilizes i9-12900H's 20 threads)
# - pin_memory=True: Speeds up CPU→GPU transfer
# - prefetch_factor=4: Preload 4 batches per worker (reduces waiting)
# - persistent_workers=True: Keep workers alive between epochs (faster)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,  # Uses weighted sampler (no shuffle)
    num_workers=12,  # Parallel data loading (12 threads)
    pin_memory=True,  # Speeds up CPU→GPU transfer
    prefetch_factor=4,  # Preload 4 batches per worker
    persistent_workers=True  # Keep workers alive between epochs
)

# Validation dataloader
# - No sampling/augmentation (evaluate on original data)
# - Larger batch size (2× training) since no gradients needed
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,  # Larger batch OK (no backprop)
    shuffle=False,  # No shuffle (consistent validation order)
    num_workers=12,
    pin_memory=True,
    prefetch_factor=4,  
    persistent_workers=True  
)

# Test dataloader (same config as validation)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=4,  
    persistent_workers=True 
)


# ============================================================
# CREATE MODEL
# ============================================================

logger.info("\nInitializing model...")

# SpatialFrequencyHybridCNN architecture:
#
# SPATIAL BRANCH:
#   Input: [B, 3, 256, 256] RGB pixels
#   1. FilterBank (15 HPF filters): [B, 3, 256, 256] → [B, 45, 256, 256]
#   2. InstanceNorm2d: Normalize filter outputs to [-3, 3]
#   3. 11 SRNet layers (2 Type 1, 5 Type 2, 4 Type 3): [B, 45, 256, 256] → [B, 256, 1, 1]
#   Output: [B, 256] spatial features
#
# DCT BRANCH:
#   Input: [B, 3, 32, 32, 8, 8] DCT coefficients
#   1. DC/AC normalization: Separate z-score norm for DC (avg) and AC (detail) coefficients
#   2. 11 SRNet layers (2 Type 1, 5 Type 2, 4 Type 3): [B, 192, 32, 32] → [B, 128, 1, 1]
#   Output: [B, 128] DCT features
#
# FUSION CLASSIFIER:
#   Input: [B, 384] (256 spatial + 128 DCT)
#   1. FC(384→256) + BN + ReLU + Dropout
#   2. FC(256→128) + BN + ReLU + Dropout
#   3. FC(128→2) + Weight clipping (±2.0)
#   Output: [B, 2] logits (clean vs stego)
#
# Key design decisions:
#   - Spatial branch has 256D output (2× DCT) because spatial patterns are more complex
#   - DCT branch has 128D output (frequency patterns are more compact)
#   - Fusion classifier uses BatchNorm1d (normalizes FC layer outputs)
#   - Final layer weights clamped to ±2.0 (prevents logit explosion)
model = SpatialFrequencyHybridCNN(
    num_channels=3,  # YCbCr channels
    target_blocks=(32, 32),  # 32×32 blocks = 256×256 pixels
    dropout=DROPOUT  # Only used in fusion classifier
)

# Move model to GPU
model = model.to(DEVICE)

# Compile model for faster execution
# if hasattr(torch, 'compile'):
#     model = torch.compile(model, mode='max-autotune')
#     logger.info("Model compiled with torch.compile()")

# ============================================================
# COUNT PARAMETERS
# ============================================================

# Count trainable vs frozen parameters
# - Trainable: All conv layers, batch norms, FC layers (~3M params)
# - Frozen: HPF filter bank (15 hand-crafted filters, ~0 params since they're fixed)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

logger.info(f"Model parameters:")
logger.info(f"  Trainable: {trainable_params:,}")
logger.info(f"  Frozen (HPF filters): {frozen_params:,}")
logger.info(f"  Total: {total_params:,}")

# ============================================================
# TRAINING SETUP
# ============================================================

# Loss function: CrossEntropyLoss for binary classification
# - Combines softmax + negative log likelihood
# - For balanced classes, expected initial loss = -log(0.5) = 0.693
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam (adaptive learning rate)
# - Better than SGD for deep networks with varying gradient scales
# - weight_decay=1e-4: L2 regularization (prevents overfitting)
# 
# Alternative: SGD with momentum (commented out below)
# - May converge slower but can generalize better
# - Requires learning rate warm-up and careful tuning
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scaler = GradScaler()
# SGD alternative (uncomment to use):
# optimizer = optim.SGD(
#     model.parameters(),
#     lr=LEARNING_RATE,
#     momentum=0.9,
#     nesterov=True,
#     weight_decay=WEIGHT_DECAY
# )

# Learning rate scheduler: CosineAnnealingLR
# - Gradually reduces LR from 1e-4 → 1e-6 over 100 epochs
# - Follows cosine curve (smooth annealing, no sudden drops)
# - Helps model converge to better minima
#
# LR schedule:
#   Epoch 0: 1e-4
#   Epoch 25: ~7e-5
#   Epoch 50: ~5e-5
#   Epoch 75: ~2e-5
#   Epoch 100: 1e-6
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,  # Anneal over all epochs
    eta_min=LEARNING_RATE / 100  # Min LR = 1e-6
)

# ============================================================
# TRAINING STATE
# ============================================================

# Track best validation accuracy for model checkpointing
best_val_acc = 0.0
best_epoch = 0
start_epoch = 0

# Early stopping: Stop training if no improvement for PATIENCE epochs
patience_counter = 0
wandb_run_id = 'h3ytg17n'

if RESUME_FROM_CHECKPOINT:
    # Check which checkpoints exist
    best_ckpt = MODEL_DIR / 'best_hybrid_model.pth'
    last_ckpt = MODEL_DIR / 'last_checkpoint.pth'
    
    checkpoint_path = None
    
    # Logic: Prefer last_checkpoint if it's newer (more recent training)
    if last_ckpt.exists() and best_ckpt.exists():
        # Compare which is more recent
        last_epoch = torch.load(last_ckpt, map_location='cpu')['epoch']
        best_epoch_ckpt = torch.load(best_ckpt, map_location='cpu')['epoch']
        
        if last_epoch > best_epoch_ckpt:
            checkpoint_path = last_ckpt
            logger.info(f"Resuming from LAST checkpoint (epoch {last_epoch}, more recent than best at {best_epoch_ckpt})...")
        else:
            checkpoint_path = best_ckpt
            logger.info(f"Resuming from BEST checkpoint (epoch {best_epoch_ckpt})...")
    elif last_ckpt.exists():
        checkpoint_path = last_ckpt
        logger.info("Resuming from LAST checkpoint...")
    elif best_ckpt.exists():
        checkpoint_path = best_ckpt
        logger.info("Resuming from BEST checkpoint...")
    
    # Load checkpoint if found
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state restored")
        
        # Restore optimizer state (critical for stable training!)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state restored")
        
        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state restored")
        
        # Restore training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        
        # Get WandB run ID for resuming
        #wandb_run_id = checkpoint.get('wandb_run_id', None)
        
        logger.info(f"\n{'='*60}")
        logger.info("CHECKPOINT RESUME SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Resuming from epoch: {start_epoch}")
        logger.info(f"Best val accuracy so far: {best_val_acc:.2f}%")
        logger.info(f"Patience counter: {patience_counter}/{PATIENCE}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"WandB run ID: {wandb_run_id}")
        logger.info(f"{'='*60}\n")
    else:
        logger.info("No checkpoint found. Starting training from scratch.")


# ============================================================
# INITIALIZE WANDB (Weights & Biases Experiment Tracking)
# ============================================================

# Init WandB
# WandB logs:
#   - Training/validation metrics (loss, accuracy)
#   - Model gradients and parameters (every 100 steps)
#   - Learning rate schedule
#   - Final test results
#   - Model checkpoint artifact
#
# View results at: https://wandb.ai/<your-username>/hybrid-spatial-frequency-steganalysis
wandb.init(
    project="hybrid-spatial-frequency-steganalysis",
    name=f"{'highrate' if USE_HIGH_RATE_DATA else 'lowrate'}_hybrid_spatial_dct",
    id=wandb_run_id,
    resume="allow" if wandb_run_id else None,
    config={
        # Model architecture
        'model_type': 'SpatialFrequencyHybridCNN',
        'spatial_layers': 11,
        'dct_layers': 11,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'dropout': DROPOUT,
        
        # Input configuration
        'target_blocks': (32, 32),
        'input_pixels': '256x256',
        'dct_channels': ['Y', 'Cb', 'Cr'],
        'num_channels': 3,
        
        # Spatial branch (HPF on pixels)
        'spatial_hpf_filters': 15,
        'spatial_filter_types': ['Kv', 'Edge', 'Kp', 'Kb', 'Gabor'],
        'spatial_architecture': 'SRNet-inspired (11 layers)',
        'spatial_normalization': 'InstanceNorm2d + TLU(T=3.0)',
        'spatial_input': '256x256 RGB pixels',
        'spatial_output': '256D feature vector',
        
        # DCT branch (frequency domain)
        'dct_architecture': 'SRNet-inspired (11 layers)',
        'dct_normalization': 'DC/AC z-score normalization',
        'dct_input': '32x32x8x8 DCT coefficients',
        'dct_output': '128D feature vector',
        
        # Fusion classifier
        'fusion_input': '384D (256 spatial + 128 DCT)',
        'fusion_architecture': 'FC(384→256)→BN→ReLU→FC(256→128)→BN→ReLU→FC(128→2)',
        'fusion_weight_clipping': '±2.0',
        'fusion_initialization': 'Xavier (gain=0.5)',
        
        # Training hyperparameters
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR',
        'scheduler_T_max': NUM_EPOCHS,
        'scheduler_eta_min': LEARNING_RATE / 100,
        'early_stop_patience': PATIENCE,
        'grad_clip_max_norm': 1.0,
        'mixed_precision': 'TF32 (automatic on Ampere GPU)',
        
        # Loss function
        'loss_function': 'CrossEntropyLoss',
        
        # Data augmentation
        'augmentation': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'flip_probability': 0.5,
        },
        
        # Stratified sampling configuration
        'stratified_sampling': {
            'enabled': True,
            'dimensions': ['clean_vs_stego', 'rgb_vs_grayscale', 'payload_rate'],
            'description': 'Each batch maintains: 50/50 clean/stego, original RGB/grayscale ratio within each class, and 25% each payload (0.1/0.2/0.3/0.4) within stego',
        },
        
        # Dataset info
        'dataset': TEST_NAME,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'total_samples': len(dataset),
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'weighted_sampling': True,
        'stratified_sampling': True,
        
        # Hardware
        'device': DEVICE,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
    },
    tags=['hybrid', 'spatial+dct', 'j-uniward', 'srnet-inspired', '11-layer-balanced', 'tf32']
)

# Watch model gradients and parameters (logged every 100 steps)
# This tracks:
#   - Gradient magnitudes (to detect explosion/vanishing)
#   - Parameter distributions (to monitor weight evolution)
#   - Layer-wise statistics (to debug training issues)
wandb.watch(model, log='all', log_freq=100)

logger.info(f"WandB initialized: {wandb.run.name} (ID: {wandb.run.id})")

# If resuming, update WandB summary with existing best stats
if RESUME_FROM_CHECKPOINT and start_epoch > 0:
    wandb.run.summary['best_val_acc'] = best_val_acc
    wandb.run.summary['best_epoch'] = start_epoch - 1


# ============================================================
# TRAINING LOOP
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: SpatialFrequencyHybridCNN model
        loader: Training dataloader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Adam optimizer
        device: 'cuda' or 'cpu'
    
    Returns:
        epoch_loss: Average loss over all batches
        epoch_acc: Average accuracy over all batches
    
    Training loop steps:
        1. Forward pass: spatial_img, dct_blocks → logits
        2. Compute loss: CrossEntropyLoss(logits, labels)
        3. Backward pass: loss.backward()
        4. Gradient clipping: max_norm=1.0 (prevents explosion)
        5. Weight clipping: Final layer weights to ±2.0 (prevents logit explosion)
        6. Optimizer step: Update model parameters
    """
    model.train()  # Enable training mode (dropout, batch norm updates)
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (spatial_img, dct_blocks, labels) in enumerate(pbar):
        # Move data to GPU
        spatial_img = spatial_img.to(device)
        dct_blocks = dct_blocks.to(device)
        labels = labels.to(device)
        
        # ========== FORWARD PASS ==========
        optimizer.zero_grad()  # Clear gradients from previous batch

        # ========== AMP FORWARD PASS ==========
        with autocast('cuda'):  # Automatic mixed precision
            outputs = model(spatial_img, dct_blocks)
            loss = criterion(outputs, labels)

        # Log first batch logits**
        if batch_idx == 0:
            logger.info(f"  First batch logits: [{outputs.min():.3f}, {outputs.max():.3f}]")
            logger.info(f"  First batch predictions: Clean={(outputs.argmax(1)==0).sum().item()}, Stego={(outputs.argmax(1)==1).sum().item()}")
        
        
        # ========== AMP BACKWARD PASS ==========
        scaler.scale(loss).backward() # Compute gradients via backpropagation
        
        # ========== GRADIENT CLIPPING ==========
        # Clip gradients to max L2 norm of 1.0
        # This prevents gradient explosion in deep networks
        # Without clipping: gradients can grow to 100+ in early epochs
        # With clipping: gradients stay bounded to ~1.0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # ========== WEIGHT CLIPPING (CRITICAL FOR STABILITY) ==========
        # Clip final layer weights to ±2.0 to prevent logit explosion
        # This was a critical bug fix - without it, logits grew to ±30+
        # 
        # Why this helps:
        #   - If weights grow large (e.g., 5.0), logits = 5.0 × features = huge
        #   - Large logits → large loss → large gradients → explosion
        #   - Clamping weights → bounded logits → stable training
        final_layer = model.fusion_classifier[-1]
        if hasattr(final_layer, 'weight'):
            with torch.no_grad():  # Don't track this operation in autograd
                final_layer.weight.clamp_(-2.0, 2.0)
        
        # ========== AMP OPTIMIZER STEP ==========
        scaler.step(optimizer)  # Update model parameters using gradients
        scaler.update()
        
        # ========== METRICS ==========
        total_loss += loss.item()
        _, predicted = outputs.max(1)  # Get class with highest logit
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar with current batch metrics
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    # ========== EPOCH METRICS ==========
    epoch_loss = total_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Validate model on validation/test set.
    
    Args:
        model: SpatialFrequencyHybridCNN model
        loader: Validation/test dataloader
        criterion: Loss function (CrossEntropyLoss)
        device: 'cuda' or 'cpu'
    
    Returns:
        val_loss: Average loss over all batches
        val_acc: Average accuracy over all batches
    
    Validation loop steps:
        1. Disable gradients: torch.no_grad() (faster, less memory)
        2. Forward pass: spatial_img, dct_blocks → logits
        3. Compute loss and accuracy
        4. No backward pass (evaluation only)
    """
    model.eval()  # Enable evaluation mode (no dropout, batch norm uses running stats)
    total_loss = 0.0
    correct = 0
    total = 0

    pred_clean = 0
    pred_stego = 0
    
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        pbar = tqdm(loader, desc="Validating", leave=False)
        for spatial_img, dct_blocks, labels in pbar:
            # Move data to GPU
            spatial_img = spatial_img.to(device)
            dct_blocks = dct_blocks.to(device)
            labels = labels.to(device)
            
            # Forward pass (no gradients tracked)
            outputs = model(spatial_img, dct_blocks)
            loss = criterion(outputs, labels)
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # **Count predictions**
            pred_clean += (predicted == 0).sum().item()
            pred_stego += (predicted == 1).sum().item()
    
    # **LOG prediction distribution
    logger.info(f"  Predictions: Clean={pred_clean} ({100*pred_clean/total:.1f}%), Stego={pred_stego} ({100*pred_stego/total:.1f}%)")


    return total_loss / len(loader), 100. * correct / total


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

logger.info("\n" + "="*60)
logger.info("STARTING TRAINING")
logger.info("="*60)

# OPTIONAL RESUME LOOP
# If told to from config at the top, will resume training from latest checkpoint
if RESUME_FROM_CHECKPOINT and (MODEL_DIR / 'best_hybrid_model.pth').exists():

    # Default to best model, but if a last_checkpoint exists switch to that
    checkpoint_path = None
    if (MODEL_DIR / 'best_hybrid_model.pth').exists():
        checkpoint_path = MODEL_DIR / 'best_hybrid_model.pth'
        logger.info("Resuming from BEST checkpoint...")
    elif (MODEL_DIR / 'last_checkpoint.pth').exists():
        checkpoint_path = MODEL_DIR / 'last_checkpoint.pth'
        logger.info("Resuming from LAST checkpoint...")

    if checkpoint_path:

        # Load checkpoint, model, and optimizer
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.load_state_dict(checkpoint['model_state_dict'])
        # Restore scheduler
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
        patience_counter = checkpoint.get('patience_counter', 0)

        logger.info(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
        logger.info(f"Patience counter: {patience_counter}/{PATIENCE}")
else:
    start_epoch = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    logger.info("-"*60)
    
    # ========== TRAIN FOR ONE EPOCH ==========
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # ========== VALIDATE ON VALIDATION SET ==========
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    # ========== GET CURRENT LEARNING RATE ==========
    current_lr = optimizer.param_groups[0]['lr']
    
    # ========== LOG TO CONSOLE ==========
    logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    logger.info(f"Learning Rate: {current_lr:.6f}")
    
    # ========== LOG TO WANDB ==========
    wandb.log({
        'epoch': epoch + 1,
        'train/loss': train_loss,
        'train/accuracy': train_acc,
        'val/loss': val_loss,
        'val/accuracy': val_acc,
        'learning_rate': current_lr,
        'best_val_acc': best_val_acc,
    })
    
    # ========== SAVE BEST MODEL ==========
    # Save checkpoint if validation accuracy improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        
        # Save model checkpoint
        model_path = MODEL_DIR / 'best_hybrid_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'wandb_run_id': wandb.run.id,
            'patience_counter': patience_counter,
        }, model_path)
        
        logger.info(f"New best model saved (val_acc={val_acc:.2f}%)")
        patience_counter = 0  # Reset patience counter
        
        # Log best model to WandB
        wandb.run.summary['best_val_acc'] = val_acc
        wandb.run.summary['best_epoch'] = epoch + 1
    else:
        patience_counter += 1
        logger.info(f"Patience: {patience_counter}/{PATIENCE}")
    
    # Save last checkpoint every N epochs (for crash recovery due to hitting system max)
    if (epoch + 1) % 5 == 0:  # Save every 5 epochs
        last_checkpoint_path = MODEL_DIR / 'last_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'wandb_run_id': wandb.run.id,
            'patience_counter': patience_counter,
        }, last_checkpoint_path)
        logger.info(f"Last checkpoint saved (epoch {epoch+1})")
    
    # ========== EARLY STOPPING ==========
    # Stop training if no improvement for PATIENCE epochs
    if patience_counter >= PATIENCE:
        logger.info(f"\n{'='*60}")
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        logger.info(f"{'='*60}")
        break
    
    # ========== STEP LEARNING RATE SCHEDULER ==========
    # Update learning rate according to cosine schedule
    scheduler.step()


# ============================================================
# FINAL EVALUATION ON TEST SET
# ============================================================

logger.info("\n" + "="*60)
logger.info("EVALUATING BEST MODEL ON TEST SET")
logger.info("="*60)

# ========== LOAD BEST MODEL CHECKPOINT ==========
checkpoint = torch.load(MODEL_DIR / 'best_hybrid_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# ========== EVALUATE ON TEST SET ==========
# Test set was NEVER seen during training (no data leakage)
# This gives an unbiased estimate of generalization performance
test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)

# ========== LOG RESULTS ==========
logger.info(f"\nTest Results:")
logger.info(f"  Loss: {test_loss:.4f}")
logger.info(f"  Accuracy: {test_acc:.2f}%")

# Log final test metrics to WandB
wandb.log({
    'test/loss': test_loss,
    'test/accuracy': test_acc,
})

wandb.run.summary['test_loss'] = test_loss
wandb.run.summary['test_acc'] = test_acc
wandb.run.summary['final_test_acc'] = test_acc

# ============================================================
# TRAINING SUMMARY
# ============================================================

logger.info("\n" + "="*60)
logger.info("TRAINING SUMMARY")
logger.info("="*60)
logger.info(f"Dataset: {TEST_NAME}")
logger.info(f"Model: SpatialFrequencyHybridCNN")
logger.info(f"Total Parameters: {trainable_params:,}")
logger.info(f"\nBest Validation Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
logger.info(f"\nModel saved to: {MODEL_DIR / 'best_hybrid_model.pth'}")
logger.info("="*60)

# ============================================================
# SAVE MODEL ARTIFACT TO WANDB
# ============================================================

logger.info("\nSaving model artifact to WandB...")

# Save model as WandB artifact for versioning and sharing
# Artifacts are immutable snapshots of models/datasets/files
model_artifact = wandb.Artifact(
    name=f"hybrid_model_{wandb.run.id}",
    type="model",
    description=f"Hybrid Spatial+DCT model - {TEST_NAME}",
    metadata={
        'test_accuracy': test_acc,
        'val_accuracy': best_val_acc,
        'epoch': best_epoch,
        'architecture': 'SpatialFrequencyHybridCNN',
        'spatial_layers': 11,
        'dct_layers': 11,
    }
)

model_artifact.add_file(str(MODEL_DIR / 'best_hybrid_model.pth'))
wandb.log_artifact(model_artifact)

logger.info(f"✓ Model artifact saved to WandB")

# Finish WandB run (uploads any remaining data)
wandb.finish()
logger.info("✓ WandB run finished")
