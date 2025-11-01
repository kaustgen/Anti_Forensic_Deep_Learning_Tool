# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose: Training script for true CNN steganography detector

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, Sampler
from pathlib import Path
import wandb
import weave
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import logging
import os
from collections import Counter

from sten_dct_cnn import StegoDCTSpatialDataset, StegDetectionCNNSpatial
from sten_dct_train import StegoDetectionTrainer  # Reuse trainer class

MODEL_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EmbeddingRateBalancedSampler(Sampler):
    """
    Sampler that ensures each epoch contains balanced samples across embedding rates.
    
    For J-UNIWARD dataset with rates: 0.1, 0.2, 0.3, 0.4 bpnzAC
    Ensures 25% of each rate per epoch.
    """
    
    def __init__(self, dataset, excel_path, batch_size, shuffle=True, seed=42):
        """
        Args:
            dataset: StegoDCTSpatialDataset or Subset
            excel_path: Path to BOSS_stego_training.xlsx (J-UNIWARD format)
            batch_size: Batch size for training
            shuffle: Whether to shuffle within each rate group
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch_counter = 0
        
        # Read Excel to get embedding rates
        df = pd.read_excel(excel_path)
        
        # Verify J-UNIWARD format
        required_columns = ['Stegnography Applied?', 'Payload (bpp AC DCT)']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Excel file must have '{col}' column (J-UNIWARD format required)")
        
        # Handle Subset (get underlying indices from original dataset)
        if hasattr(dataset, 'indices'):
            # This is a Subset - map position in subset to original dataset index
            original_indices = dataset.indices
            self.is_subset = True
        else:
            original_indices = list(range(len(dataset)))
            self.is_subset = False
        
        # Group SUBSET POSITIONS by embedding rate (not original indices!)
        self.rate_groups = {
            0.1: [],
            0.2: [],
            0.3: [],
            0.4: [],
            'clean': []
        }
        
        # Iterate through subset positions
        for subset_position, original_idx in enumerate(original_indices):
            row = df.iloc[original_idx]
            
            # Check if stego
            if row['Stegnography Applied?']:
                rate = float(row['Payload (bpp AC DCT)'])
                
                # Round to nearest rate (handle floating point errors)
                if abs(rate - 0.1) < 0.05:
                    self.rate_groups[0.1].append(subset_position)
                elif abs(rate - 0.2) < 0.05:
                    self.rate_groups[0.2].append(subset_position)
                elif abs(rate - 0.3) < 0.05:
                    self.rate_groups[0.3].append(subset_position)
                elif abs(rate - 0.4) < 0.05:
                    self.rate_groups[0.4].append(subset_position)
                else:
                    logger.warning(f"Unexpected embedding rate: {rate} at original index {original_idx}")
            else:
                self.rate_groups['clean'].append(subset_position)
        
        # Verify we have samples for all rates
        rate_counts = [len(self.rate_groups[rate]) for rate in [0.1, 0.2, 0.3, 0.4]]
        clean_count = len(self.rate_groups['clean'])
        
        if min(rate_counts) == 0:
            raise ValueError(
                f"Missing samples for some embedding rates! Counts: "
                f"0.1={rate_counts[0]}, 0.2={rate_counts[1]}, "
                f"0.3={rate_counts[2]}, 0.4={rate_counts[3]}"
            )
        
        # Strategy: Balance CLASSES (50% clean, 50% stego), with stego evenly distributed across rates
        # Use ALL available samples
        
        # For balanced sampling:
        # - Use min(clean_count, min(rate_counts) * 4) samples per epoch
        # - This ensures we never oversample any group
        min_rate_count = min(rate_counts)
        max_possible_stego = min_rate_count * 4  # 4 rates, each contributes equally
        
        # Use the smaller of clean or total stego to ensure balance
        samples_per_class = min(clean_count, max_possible_stego)
        
        self.samples_per_rate = samples_per_class // 4  # Split evenly across 4 rates
        self.samples_clean = samples_per_class
        self.total_epoch_size = self.samples_clean + (self.samples_per_rate * 4)
        
        # Log distribution
        total_stego_per_epoch = self.samples_per_rate * 4
        logger.info("Embedding rate distribution in sampler:")
        logger.info(f"  Clean: {clean_count} available → {self.samples_clean} per epoch (50.0% of epoch)")
        for rate in [0.1, 0.2, 0.3, 0.4]:
            count = len(self.rate_groups[rate])
            pct_of_epoch = 12.5  # Each rate is 12.5% of total epoch
            usage_pct = 100 * self.samples_per_rate / count if count > 0 else 0
            logger.info(f"  {rate} bpnzAC: {count} available → {self.samples_per_rate} per epoch ({pct_of_epoch}% of epoch, {usage_pct:.1f}% used)")
        
        logger.info(f"\nBalanced sampler configuration:")
        logger.info(f"  Clean samples per epoch: {self.samples_clean} (50%)")
        logger.info(f"  Stego samples per epoch: {total_stego_per_epoch} (50%)")
        logger.info(f"  Total epoch size: {self.total_epoch_size}")
        logger.info(f"  Batches per epoch: {self.total_epoch_size // batch_size}")
        logger.info(f"  Class balance: 50% clean, 50% stego")
    
    def __iter__(self):
        """
        Generate indices for one epoch with balanced rates.
        
        IMPORTANT: Returns indices relative to the Subset (0 to len(subset)-1),
        NOT original dataset indices. This is correct for DataLoader usage.
        """
        
        # Set random seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch_counter)
        
        epoch_indices = []
        
        # Sample equal amounts from each rate
        for rate in [0.1, 0.2, 0.3, 0.4]:
            rate_indices = self.rate_groups[rate].copy()
            
            if self.shuffle:
                perm = torch.randperm(len(rate_indices), generator=g).tolist()
                rate_indices = [rate_indices[i] for i in perm]
            
            # Take exactly samples_per_rate
            rate_indices = rate_indices[:self.samples_per_rate]
            epoch_indices.extend(rate_indices)
        
        # Add clean samples
        clean_indices = self.rate_groups['clean'].copy()
        if self.shuffle:
            perm = torch.randperm(len(clean_indices), generator=g).tolist()
            clean_indices = [clean_indices[i] for i in perm]
        
        # Take exactly samples_clean
        clean_indices = clean_indices[:self.samples_clean]
        epoch_indices.extend(clean_indices)
        
        # Final shuffle of all samples together
        if self.shuffle:
            perm = torch.randperm(len(epoch_indices), generator=g).tolist()
            epoch_indices = [epoch_indices[i] for i in perm]
        
        # Increment epoch counter for next call
        self.epoch_counter += 1

        # Verify 50/50 class balance
        stego_indices_set = set()
        for rate in [0.1, 0.2, 0.3, 0.4]:
            stego_indices_set.update(self.rate_groups[rate])
        
        clean_count = sum(1 for idx in epoch_indices if idx not in stego_indices_set)
        stego_count = len(epoch_indices) - clean_count
        
        # Verify balance (allow small deviation due to rounding)
        balance_ratio = clean_count / stego_count if stego_count > 0 else 0
        
        if abs(balance_ratio - 1.0) > 0.05:  # Allow 5% deviation
            logger.warning(
                f"Epoch {self.epoch_counter} class imbalance detected! "
                f"Clean={clean_count} ({100*clean_count/len(epoch_indices):.1f}%), "
                f"Stego={stego_count} ({100*stego_count/len(epoch_indices):.1f}%)"
            )
        
        return iter(epoch_indices)
    
    def __len__(self):
        """Return total samples per epoch"""
        return self.total_epoch_size


class AugmentedSubset(Subset):
    """Wrapper around Subset that enables augmentation for training data."""
    def __init__(self, dataset, indices, train=False):
        super().__init__(dataset, indices)
        self.train_mode = train
        # Save original train flag and set it
        self.original_train = getattr(dataset, 'train', False)
        
    def __getitem__(self, idx):
        # Temporarily set train mode on underlying dataset
        if hasattr(self.dataset, 'train'):
            old_train = self.dataset.train
            self.dataset.train = self.train_mode
            result = super().__getitem__(idx)
            self.dataset.train = old_train
            return result
        return super().__getitem__(idx)


@weave.op()
def stratified_split_with_rates(dataset, excel_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Perform stratified split that preserves:
    1. Class balance (clean vs stego)
    2. Embedding rate distribution (0.1, 0.2, 0.3, 0.4)
    
    Args:
        dataset: Dataset with labels
        excel_path: Path to Excel file with embedding rates (J-UNIWARD format)
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as Subset objects
    """
    # Read Excel to get labels and rates
    df = pd.read_excel(excel_path)
    
    # Verify J-UNIWARD format
    if 'Payload (bpp AC DCT)' not in df.columns:
        raise ValueError("Excel must have 'Payload (bpp AC DCT)' column (J-UNIWARD format required)")
    
    # Create stratification key: "clean" or "stego_0.1", "stego_0.2", etc.
    strat_keys = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        if row['Stegnography Applied?']:
            rate = float(row['Payload (bpp AC DCT)'])
            
            # Round to nearest rate
            if abs(rate - 0.1) < 0.05:
                strat_keys.append('stego_0.1')
            elif abs(rate - 0.2) < 0.05:
                strat_keys.append('stego_0.2')
            elif abs(rate - 0.3) < 0.05:
                strat_keys.append('stego_0.3')
            elif abs(rate - 0.4) < 0.05:
                strat_keys.append('stego_0.4')
            else:
                raise ValueError(f"Unexpected embedding rate: {rate} at index {idx}")
        else:
            strat_keys.append('clean')
    
    strat_keys = np.array(strat_keys)
    indices = np.arange(len(dataset))
    
    # Log distribution
    unique, counts = np.unique(strat_keys, return_counts=True)
    logger.info("Dataset stratification groups:")
    for key, count in zip(unique, counts):
        logger.info(f"  {key}: {count} samples ({100*count/len(dataset):.1f}%)")
    
    # Split train vs (val+test) with stratification
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=strat_keys,
        random_state=seed
    )
    
    # Split val vs test
    temp_strat = strat_keys[temp_indices]
    val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_of_temp,
        stratify=temp_strat,
        random_state=seed
    )
    
    # Log split statistics
    logger.info("\nStratified split by embedding rate:")
    logger.info(f"  Train: {len(train_indices)} samples")
    for key in unique:
        count = np.sum(strat_keys[train_indices] == key)
        logger.info(f"    {key}: {count} ({100*count/len(train_indices):.1f}%)")
    
    logger.info(f"  Val: {len(val_indices)} samples")
    for key in unique:
        count = np.sum(strat_keys[val_indices] == key)
        logger.info(f"    {key}: {count} ({100*count/len(val_indices):.1f}%)")
    
    logger.info(f"  Test: {len(test_indices)} samples")
    for key in unique:
        count = np.sum(strat_keys[test_indices] == key)
        logger.info(f"    {key}: {count} ({100*count/len(test_indices):.1f}%)")
    
    # Create subsets with augmentation enabled for training
    train_dataset = AugmentedSubset(dataset, train_indices.tolist(), train=True)
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    
    return train_dataset, val_dataset, test_dataset


@weave.op()
def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Perform stratified split of dataset to maintain class balance across splits.
    
    Args:
        dataset: Dataset with labels accessible via dataset[idx][1]
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as Subset objects
    """
    # Extract all labels
    logger.info("Extracting labels for stratified split...")
    labels = []
    for idx in range(len(dataset)):
        try:
            _, label = dataset[idx]
            labels.append(label)
        except Exception as e:
            logger.warning("Failed to load sample %d for stratification: %s", idx, e)
            labels.append(-1)  # Mark as invalid
    
    labels = np.array(labels)
    indices = np.arange(len(dataset))
    
    # Filter out invalid samples
    valid_mask = labels >= 0
    valid_indices = indices[valid_mask]
    valid_labels = labels[valid_mask]
    
    if len(valid_indices) == 0:
        raise RuntimeError("No valid samples found in dataset")
    
    # Count class distribution
    unique, counts = np.unique(valid_labels, return_counts=True)
    logger.info("Overall class distribution:")
    for cls, count in zip(unique, counts):
        logger.info("  Class %d: %d samples (%.1f%%)", cls, count, 100 * count / len(valid_labels))
    
    # First split: train vs (val+test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        valid_indices,
        valid_labels,
        train_size=train_ratio,
        stratify=valid_labels,
        random_state=seed
    )
    
    # Second split: val vs test (from temp)
    # Calculate proportion: if we have 0.15 val and 0.15 test from total,
    # and temp is 0.30 of total, then val should be 0.15/0.30 = 0.5 of temp
    val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices,
        temp_labels,
        train_size=val_ratio_of_temp,
        stratify=temp_labels,
        random_state=seed
    )
    
    # Log split statistics
    logger.info("Stratified split complete:")
    logger.info("  Train: %d samples", len(train_indices))
    for cls in unique:
        count = np.sum(train_labels == cls)
        logger.info("    Class %d: %d (%.1f%%)", cls, count, 100 * count / len(train_labels))
    
    logger.info("  Val: %d samples", len(val_indices))
    for cls in unique:
        count = np.sum(val_labels == cls)
        logger.info("    Class %d: %d (%.1f%%)", cls, count, 100 * count / len(val_labels))
    
    logger.info("  Test: %d samples", len(test_indices))
    for cls in unique:
        count = np.sum(test_labels == cls)
        logger.info("    Class %d: %d (%.1f%%)", cls, count, 100 * count / len(test_labels))
    
    # Create Subset datasets with augmentation enabled for training
    train_dataset = AugmentedSubset(dataset, train_indices.tolist(), train=True)
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    
    return train_dataset, val_dataset, test_dataset


@weave.op()
def main():
    """Main training pipeline for spatial CNN"""
    
    # Configuration
    EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'BOSS_stego_training.xlsx'
    #EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'
    IMG_ROOT = Path('.')
    BATCH_SIZE = 64  # Increased for better gradient estimates with SGD
    EPOCHS = 100  # Reduced from 250 (BOSS is larger dataset)
    EARLY_STOP_PATIENCE = 20  # Increased from 15 for low-payload learning
    LEARNING_RATE = 1e-3  # SGD needs higher LR 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TARGET_BLOCKS = (32, 32)  # 32×32 blocks = 256×256 pixels
    USE_MIXUP = False  # Disable mixup initially - may be confusing the model
    MIXUP_ALPHA = 0.2  # Moderate mixing strength
    
    # Initialize Weave
    weave.init('stego-detection-cnn')
    
    if DEVICE.startswith('cuda'):
        try:
            torch.backends.cudnn.benchmark = True
            logger.info('Enabled torch.backends.cudnn.benchmark')
        except Exception:
            pass
    
    logger.info("Using device: %s", DEVICE)
    logger.info("Target block size: %s (=%d×%d pixels)", TARGET_BLOCKS, TARGET_BLOCKS[0]*8, TARGET_BLOCKS[1]*8)
    logger.info("Training config: BATCH_SIZE=%d, LR=%.1e, EPOCHS=%d, MIXUP=%s", 
                BATCH_SIZE, LEARNING_RATE, EPOCHS, USE_MIXUP)
    
    # Create dataset
    dataset = StegoDCTSpatialDataset(
        EXCEL_PATH,
        IMG_ROOT,
        dct_channels=['Y', 'Cb', 'Cr'],
        target_blocks=TARGET_BLOCKS
    )
    
    logger.info("Dataset size: %d images", len(dataset))
    
    # Get a sample to verify dimensions
    try:
        sample_features, sample_label = dataset[0]
        logger.info("Sample feature shape: %s, label: %d", sample_features.shape, sample_label)
        
        # Check if sample has zero-padded channels (grayscale image)
        for c_idx in range(sample_features.shape[0]):
            channel_sum = sample_features[c_idx].abs().sum().item()
            if channel_sum < 1e-6:
                logger.info("Channel %d appears to be zero-padded (grayscale image)", c_idx)
        
        expected_shape = (len(dataset.dct_channels), *TARGET_BLOCKS, 8, 8)
        if tuple(sample_features.shape) != expected_shape:
            logger.warning("Unexpected shape! Expected %s, got %s", expected_shape, sample_features.shape)
    except Exception as e:
        logger.error("Failed to load sample: %s", e)
        return
    
    # Analyze dataset composition (grayscale vs color)
    logger.info("Analyzing dataset composition...")
    grayscale_count = 0
    color_count = 0
    sample_size = min(100, len(dataset))  # Sample 100 images to check
    
    for idx in range(0, len(dataset), len(dataset) // sample_size):
        try:
            features, _ = dataset[idx]
            # Check if Cb/Cr channels (1 and 2) are all zeros
            if features[1:].abs().sum() < 1e-6:
                grayscale_count += 1
            else:
                color_count += 1
        except Exception:
            continue
    
    total_sampled = grayscale_count + color_count
    if total_sampled > 0:
        grayscale_pct = 100.0 * grayscale_count / total_sampled
        color_pct = 100.0 * color_count / total_sampled
        logger.info("Dataset composition (sampled %d images):", total_sampled)
        logger.info("  Grayscale: %d (%.1f%%)", grayscale_count, grayscale_pct)
        logger.info("  Color: %d (%.1f%%)", color_count, color_pct)

    

    # Stratified split WITH RATE PRESERVATION
    train_dataset, val_dataset, test_dataset = stratified_split_with_rates(
        dataset,
        excel_path=EXCEL_PATH,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Compute class weights from training set to combat class imbalance
    logger.info("Computing class weights from training set...")
    train_labels = []
    for idx in train_dataset.indices:
        _, label = dataset[idx]
        train_labels.append(label)
    
    label_counts = Counter(train_labels)
    total_train = len(train_labels)
    
    # Inverse frequency weighting
    class_weights = torch.tensor([
        total_train / (2 * label_counts[0]),  # Clean (class 0)
        total_train / (2 * label_counts[1])   # Stego (class 1)
    ], dtype=torch.float32).to(DEVICE)
    
    logger.info("Class distribution in training set:")
    logger.info("  Clean (0): %d samples (%.1f%%)", label_counts[0], 100 * label_counts[0] / total_train)
    logger.info("  Stego (1): %d samples (%.1f%%)", label_counts[1], 100 * label_counts[1] / total_train)
    logger.info("Class weights: Clean=%.3f, Stego=%.3f", class_weights[0].item(), class_weights[1].item())
    
    # Balance test set to ensure fair evaluation
    logger.info("Balancing test set for fair evaluation...")
    test_indices = test_dataset.indices
    test_labels = []
    for idx in test_indices:
        _, label = dataset[idx]
        test_labels.append(label)
    
    test_clean_indices = [idx for idx, label in zip(test_indices, test_labels) if label == 0]
    test_stego_indices = [idx for idx, label in zip(test_indices, test_labels) if label == 1]
    
    # Use minimum count to balance
    min_test_samples = min(len(test_clean_indices), len(test_stego_indices))
    
    import random
    random.seed(42)
    balanced_test_indices = (
        random.sample(test_clean_indices, min_test_samples) +
        random.sample(test_stego_indices, min_test_samples)
    )
    
    test_dataset = Subset(dataset, balanced_test_indices)
    logger.info("Balanced test set: %d Clean + %d Stego = %d total", 
                min_test_samples, min_test_samples, len(test_dataset))
    
    # Create balanced samplers for training and validation
    logger.info("Creating embedding rate balanced samplers...")
    train_sampler = EmbeddingRateBalancedSampler(
        dataset=train_dataset,
        excel_path=EXCEL_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    
    val_sampler = EmbeddingRateBalancedSampler(
        dataset=val_dataset,
        excel_path=EXCEL_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't shuffle validation
        seed=42
    )
    
    # Create dataloaders
    num_workers = int(os.environ.get('NUM_WORKERS', '0'))
    pin_memory = True if DEVICE.startswith('cuda') else False
    
    logger.info('DataLoader settings: num_workers=%d, pin_memory=%s, batch_size=%d', 
               num_workers, pin_memory, BATCH_SIZE)
    
    # Create dataloaders with balanced samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    logger.info("Dataset splits: Train=%d Val=%d Test=%d", 
                len(train_sampler), len(val_sampler), len(test_dataset))
    logger.info("Batches per epoch: Train=%d Val=%d Test=%d",
                len(train_sampler) // BATCH_SIZE,
                len(val_sampler) // BATCH_SIZE,
                len(test_loader))
    
    model = StegDetectionCNNSpatial(
        num_channels=len(dataset.dct_channels),
        target_blocks=TARGET_BLOCKS,
        dropout=0.2  # Reduced from 0.4 - too much regularization prevents learning
    )
    model_name = "CNNSpatial"
    
    logger.info("Model architecture: %s\n%s", model_name, model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %d (trainable: %d)", total_params, trainable_params)
    
    # Initialize trainer (reuse from sten_dct_train)
    trainer = StegoDetectionTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights
    )
    
    # Initialize wandb AFTER trainer creation so we can access optimizer/loss details
    wandb_run = None
    if wandb is not None:
        try:
            # Extract trainer config
            optimizer_name = trainer.optimizer.__class__.__name__
            loss_fn_name = trainer.criterion.__class__.__name__
            
            # Get optimizer hyperparameters
            optimizer_config = {
                'type': optimizer_name,
                'lr': LEARNING_RATE,
            }
            
            # Add optimizer-specific params
            if hasattr(trainer.optimizer, 'weight_decay'):
                optimizer_config['weight_decay'] = trainer.optimizer.param_groups[0]['weight_decay']
            if hasattr(trainer.optimizer, 'momentum'):
                optimizer_config['momentum'] = trainer.optimizer.param_groups[0].get('momentum', 0)
            if hasattr(trainer.optimizer, 'betas'):
                optimizer_config['betas'] = trainer.optimizer.param_groups[0].get('betas', (0.9, 0.999))
            
            # Loss function config
            loss_config = {
                'type': loss_fn_name,
            }
            if hasattr(trainer.criterion, 'smoothing'):
                loss_config['label_smoothing'] = trainer.criterion.smoothing
            
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'stegnography-detector-cnn'),
                name=os.environ.get('WANDB_RUN_NAME', f'spatial_cnn_{model_name}'),
                config={
                    # Model architecture
                    'model': model_name,
                    'model_type': 'StegDetectionCNNSpatial',
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'dropout': 0.4,  # Increased for better regularization
                    
                    # Input configuration
                    'target_blocks': TARGET_BLOCKS,
                    'input_shape': f'{TARGET_BLOCKS[0]*8}x{TARGET_BLOCKS[1]*8}',
                    'dct_channels': dataset.dct_channels,
                    'num_channels': len(dataset.dct_channels),
                    
                    # Training hyperparameters
                    'batch_size': BATCH_SIZE,
                    'epochs': EPOCHS,
                    'early_stop_patience': EARLY_STOP_PATIENCE,
                    'learning_rate': LEARNING_RATE,
                    
                    # Optimizer configuration
                    'optimizer': optimizer_config,
                    
                    # Loss function
                    'loss_function': loss_config,
                    
                    # Data augmentation
                    'augmentation': {
                        'horizontal_flip': True,
                        'vertical_flip': True,
                        'flip_probability': 0.5,
                        'mixup': USE_MIXUP,
                        'mixup_alpha': MIXUP_ALPHA,
                    },
                    
                    # Dataset info
                    'dataset': 'Custom',
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset),
                    'test_samples': len(test_dataset),
                    'total_samples': len(dataset),
                    'grayscale_pct': grayscale_pct if total_sampled > 0 else 0,
                    'color_pct': color_pct if total_sampled > 0 else 0,
                    
                    # Hardware
                    'device': DEVICE,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                }
            )
            wandb_run = wandb.run
            wandb.watch(model, log='all', log_freq=100)
            logger.info('Initialized wandb run: %s', getattr(wandb_run, 'name', None))
        except Exception:
            logger.exception('wandb.init failed; continuing without wandb')
    
    # Train
    logger.info("Starting training with mixup augmentation (alpha=%.2f)...", MIXUP_ALPHA)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        use_mixup=USE_MIXUP,
        mixup_alpha=MIXUP_ALPHA
    )
    
    # Plot training history
    trainer.plot_training_history(save_path=f'training_history_{model_name}.png')
    
    # Evaluate
    results = trainer.evaluate(test_loader, save_plots=True)
    
    # Log test metrics to wandb
    if wandb is not None and wandb_run is not None:
        try:
            wandb.log({
                'test_loss': results['test_loss'],
                'test_accuracy': results['test_acc'],
                'final_test_acc': results['test_acc'],  # Also log with explicit name
            })
            logger.info("Logged test metrics to wandb")
        except Exception:
            logger.exception('Failed to log test metrics to wandb')
    
    # Save final model
    final_path = MODEL_DIR / f'final_stego_model_{model_name}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_channels': len(dataset.dct_channels),
            'target_blocks': TARGET_BLOCKS,
            'model_type': model_name,
        },
        'results': results
    }, str(final_path))
    logger.info("Saved final model to %s", final_path)
    
    if wandb is not None and wandb_run is not None:
        try:
            wandb.save(str(final_path))
            wandb.finish()
        except Exception:
            logger.exception('wandb.save/finish failed')


if __name__ == "__main__":
    main()
