# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose: Training script for true CNN steganography detector

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import logging
import os

from sten_dct_cnn import StegoDCTSpatialDataset, StegDetectionCNNSpatial
from sten_dct_train import StegoDetectionTrainer  # Reuse trainer class

MODEL_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    
    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training pipeline for spatial CNN"""
    
    # Configuration
    EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'
    IMG_ROOT = Path('.')
    BATCH_SIZE = 16  # Can be larger since CNN processes spatially
    EPOCHS = 250
    EARLY_STOP_PATIENCE = 10
    LEARNING_RATE = 1e-2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TARGET_BLOCKS = (32, 32)  # 32×32 blocks = 256×256 pixels
    
    if DEVICE.startswith('cuda'):
        try:
            torch.backends.cudnn.benchmark = True
            logger.info('Enabled torch.backends.cudnn.benchmark')
        except Exception:
            pass
    
    logger.info("Using device: %s", DEVICE)
    logger.info("Target block size: %s (=%d×%d pixels)", TARGET_BLOCKS, TARGET_BLOCKS[0]*8, TARGET_BLOCKS[1]*8)
    
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
        expected_shape = (len(dataset.dct_channels), *TARGET_BLOCKS, 8, 8)
        if tuple(sample_features.shape) != expected_shape:
            logger.warning("Unexpected shape! Expected %s, got %s", expected_shape, sample_features.shape)
    except Exception as e:
        logger.error("Failed to load sample: %s", e)
        return
    

    # Stratified split to maintain class balance
    train_dataset, val_dataset, test_dataset = stratified_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Create dataloaders
    num_workers = int(os.environ.get('NUM_WORKERS', '0'))
    pin_memory = True if DEVICE.startswith('cuda') else False
    
    logger.info('DataLoader settings: num_workers=%d, pin_memory=%s, batch_size=%d', 
               num_workers, pin_memory, BATCH_SIZE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
    
    logger.info("Dataset splits: Train=%d Val=%d Test=%d", len(train_dataset), len(val_dataset), len(test_dataset))
    
    model = StegDetectionCNNSpatial(
        num_channels=len(dataset.dct_channels),
        target_blocks=TARGET_BLOCKS,
        dropout=0.5
    )
    model_name = "CNNSpatial"
    
    logger.info("Model architecture: %s\n%s", model_name, model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %d (trainable: %d)", total_params, trainable_params)
    
    # Initialize wandb
    wandb_run = None
    if wandb is not None:
        try:
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'stegnography-detector-cnn'),
                name=os.environ.get('WANDB_RUN_NAME', f'spatial_cnn_{model_name}'),
                config={
                    'model': model_name,
                    'batch_size': BATCH_SIZE,
                    'epochs': EPOCHS,
                    'learning_rate': LEARNING_RATE,
                    'device': DEVICE,
                    'target_blocks': TARGET_BLOCKS,
                    'dct_channels': dataset.dct_channels,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                }
            )
            wandb_run = wandb.run
            wandb.watch(model, log='all', log_freq=100)
            logger.info('Initialized wandb run: %s', getattr(wandb_run, 'name', None))
        except Exception:
            logger.exception('wandb.init failed; continuing without wandb')
    
    # Initialize trainer (reuse from sten_dct_train)
    trainer = StegoDetectionTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE
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
