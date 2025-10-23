#!/usr/bin/env python3
"""
test_model_by_payload.py

Test a trained steganography detection model against images filtered by payload size.
Allows filtering by payload threshold (e.g., only test images with payload < 0.5 bpp AC DCT).

Adjust the global variables in main() to configure:
    - MIN_PAYLOAD: Minimum payload threshold (None for no minimum)
    - MAX_PAYLOAD: Maximum payload threshold (None for no maximum)
    - MODEL_PATH: Path to trained model checkpoint
    - EXCEL_PATH: Path to Excel with payload statistics
    - BATCH_SIZE: Batch size for evaluation
    - DEVICE: 'cuda', 'cpu', or None for auto-detect
    - OUTPUT_DIR: Directory to save results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architectures
import sys
sys.path.append(str(Path(__file__).parent.parent))
from sten_dct_cnn import StegoDCTSpatialDataset, StegDetectionCNNSpatial
from sten_dct_train import StegoDetectionTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_dataset_by_payload(excel_path, min_payload=None, max_payload=None, payload_column='Payload (bpp AC DCT)'):
    """
    Filter dataset indices based on payload size.
    Returns balanced dataset: stego images matching payload criteria + equal number of randomly sampled clean images.
    
    Args:
        excel_path: Path to Excel with payload statistics
        min_payload: Minimum payload threshold (inclusive), None for no minimum
        max_payload: Maximum payload threshold (inclusive), None for no maximum
        payload_column: Column name containing payload values
    
    Returns:
        tuple: (filtered_indices, stats_dict)
    """
    # Load Excel
    df = pd.read_excel(excel_path)
    
    if payload_column not in df.columns:
        raise ValueError(f"Column '{payload_column}' not found in Excel. Available: {list(df.columns)}")
    
    # Get stego images only (clean images have N/A payload)
    stego_mask = df.iloc[:, 1] == True  # Second column is "Stegnography Applied?"
    
    # First pass: collect stego images matching payload criteria
    stego_indices = []
    payload_values = []
    
    for idx in range(len(df)):
        if not stego_mask.iloc[idx]:
            continue  # Skip clean images for now
        
        # For stego images, check payload threshold
        payload_val = df[payload_column].iloc[idx]
        
        # Handle N/A or ERROR values
        if pd.isna(payload_val) or payload_val == 'N/A' or payload_val == 'ERROR' or payload_val == 'NO COVER':
            continue
        
        try:
            payload_float = float(payload_val)
        except (ValueError, TypeError):
            continue
        
        # Apply filters
        if min_payload is not None and payload_float < min_payload:
            continue
        if max_payload is not None and payload_float > max_payload:
            continue
        
        stego_indices.append(idx)
        payload_values.append(payload_float)
    
    # Second pass: get all clean image indices
    clean_indices = [idx for idx in range(len(df)) if not stego_mask.iloc[idx]]
    
    # Randomly sample clean images to match the number of filtered stego images
    num_stego = len(stego_indices)
    num_clean_available = len(clean_indices)
    
    if num_clean_available > 0:
        num_clean_to_sample = min(num_stego, num_clean_available)
        np.random.seed(42)  # For reproducibility
        sampled_clean_indices = np.random.choice(clean_indices, size=num_clean_to_sample, replace=False).tolist()
    else:
        sampled_clean_indices = []
        num_clean_to_sample = 0
    
    # Combine stego and clean indices
    filtered_indices = stego_indices + sampled_clean_indices
    
    # Calculate statistics
    stats = {
        'total_images': len(df),
        'total_stego': stego_mask.sum(),
        'total_clean': (~stego_mask).sum(),
        'filtered_count': len(filtered_indices),
        'filtered_clean': num_clean_to_sample,
        'filtered_stego': num_stego,
        'min_payload': np.min(payload_values) if payload_values else None,
        'max_payload': np.max(payload_values) if payload_values else None,
        'mean_payload': np.mean(payload_values) if payload_values else None,
        'median_payload': np.median(payload_values) if payload_values else None,
        'std_payload': np.std(payload_values) if payload_values else None,
    }
    
    return filtered_indices, stats


def evaluate_filtered_subset(model, dataset, filtered_indices, device='cuda', batch_size=16):
    """
    Evaluate model on filtered subset of dataset.
    
    Args:
        model: Trained model
        dataset: Full dataset
        filtered_indices: List of indices to evaluate
        device: Device to run on
        batch_size: Batch size for evaluation
    
    Returns:
        dict with evaluation results
    """
    # Create subset
    subset = Subset(dataset, filtered_indices)
    
    logger.info("Evaluating on %d filtered images...", len(subset))
    
    # Create dataloader
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.startswith('cuda') else False
    )
    
    # Evaluate
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except:
        auc_score = None
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'labels': all_labels,
        'predictions': all_predictions,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'auc': auc_score
    }


def plot_results(results, output_dir, prefix='filtered'):
    """Generate plots for filtered evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Clean', 'Stego'],
        yticklabels=['Clean', 'Stego']
    )
    plt.title(f'Confusion Matrix - {prefix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / f'{prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    if results['auc'] is not None:
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {results['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {prefix}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f'{prefix}_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # ==================== CONFIGURATION ====================
    # Adjust these variables to configure the test
    
    # Payload filtering (bpp AC DCT)
    MIN_PAYLOAD = None      # Minimum payload threshold (None for no minimum)
    MAX_PAYLOAD = 0.2       # Maximum payload threshold (None for no maximum)
    
    # File paths
    MODEL_PATH = 'best_stego_model.pth'                    # Path to trained model
    EXCEL_PATH = '../../dataGen/stego_training.xlsx'       # Path to Excel with payload stats
    OUTPUT_DIR = 'filtered_test_results'                   # Output directory for results
    
    # Training parameters
    BATCH_SIZE = 16         # Batch size for evaluation
    DEVICE = None           # 'cuda', 'cpu', or None for auto-detect
    
    # =======================================================
    
    # Setup device
    if DEVICE is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = DEVICE
    
    logger.info("Using device: %s", device)
    
    # Convert paths to absolute
    base_dir = Path(__file__).parent
    model_path = Path(MODEL_PATH)
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    
    excel_path = Path(EXCEL_PATH)
    if not excel_path.is_absolute():
        excel_path = base_dir / excel_path
    
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    
    # Check files exist
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return
    
    if not excel_path.exists():
        logger.error("Excel not found: %s", excel_path)
        return
    
    logger.info("="*60)
    logger.info("PAYLOAD-FILTERED MODEL TESTING")
    logger.info("="*60)
    logger.info("Model: %s", model_path)
    logger.info("Dataset: %s", excel_path)
    logger.info("Payload filter: min=%s, max=%s", MIN_PAYLOAD, MAX_PAYLOAD)
    logger.info("="*60)
    
    # Filter dataset by payload
    logger.info("Filtering dataset by payload...")
    filtered_indices, stats = filter_dataset_by_payload(
        excel_path,
        min_payload=MIN_PAYLOAD,
        max_payload=MAX_PAYLOAD
    )
    
    logger.info("Dataset statistics:")
    logger.info("  Total images: %d", stats['total_images'])
    logger.info("  Total clean images: %d", stats['total_clean'])
    logger.info("  Total stego images: %d", stats['total_stego'])
    logger.info("  Filtered images: %d (Clean: %d, Stego: %d)", stats['filtered_count'], stats['filtered_clean'], stats['filtered_stego'])
    logger.info("  Stego payload range: [%.6f, %.6f]", stats['min_payload'] or 0, stats['max_payload'] or 0)
    logger.info("  Mean payload: %.6f", stats['mean_payload'] or 0)
    logger.info("  Median payload: %.6f", stats['median_payload'] or 0)
    logger.info("  Std payload: %.6f", stats['std_payload'] or 0)
    
    if len(filtered_indices) == 0:
        logger.error("No images match the payload filter criteria!")
        return
    
    # Load model
    logger.info("Loading model from %s...", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model type and config
    if 'model_config' in checkpoint:
        # Spatial CNN with metadata
        config = checkpoint['model_config']
        model = StegDetectionCNNSpatial(
            num_channels=config.get('num_channels', 3),
            target_blocks=config.get('target_blocks', (32, 32)),
            dropout=0.5
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        dataset_type = 'spatial'
        logger.info("Detected Spatial CNN model with metadata")
    elif 'model_state_dict' in checkpoint:
        # Checkpoint has separate state dict
        config = {'num_channels': 3, 'target_blocks': (32, 32)}
        model = StegDetectionCNNSpatial(
            num_channels=config['num_channels'],
            target_blocks=config['target_blocks'],
            dropout=0.5
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        dataset_type = 'spatial'
        logger.info("Detected Spatial CNN model (using default config)")
    else:
        # Direct state dict (from torch.save(model.state_dict(), ...))
        config = {'num_channels': 3, 'target_blocks': (32, 32)}
        model = StegDetectionCNNSpatial(
            num_channels=config['num_channels'],
            target_blocks=config['target_blocks'],
            dropout=0.5
        )
        model.load_state_dict(checkpoint)
        dataset_type = 'spatial'
        logger.info("Detected Spatial CNN model (direct state_dict, using default config)")
    
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Create dataset
    logger.info("Loading dataset...")
    if dataset_type == 'spatial':
        dataset = StegoDCTSpatialDataset(
            excel_path,
            Path('.'),
            dct_channels=['Y', 'Cb', 'Cr'],
            target_blocks=config.get('target_blocks', (32, 32))
        )
    
    logger.info("Dataset loaded: %d total images", len(dataset))
    
    # Evaluate on filtered subset
    logger.info("="*60)
    logger.info("Evaluating model on filtered subset...")
    logger.info("="*60)
    
    results = evaluate_filtered_subset(
        model,
        dataset,
        filtered_indices,
        device=device,
        batch_size=BATCH_SIZE
    )
    
    # Print results
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info("Loss: %.4f", results['loss'])
    logger.info("Accuracy: %.2f%%", results['accuracy'])
    if results['auc'] is not None:
        logger.info("AUC: %.4f", results['auc'])
    logger.info("="*60)
    
    # Classification report
    report = classification_report(
        results['labels'],
        results['predictions'],
        labels=[0, 1],
        target_names=['Clean', 'Steganography'],
        zero_division=0
    )
    logger.info("Classification Report:\n%s", report)
    
    # Confusion matrix
    logger.info("Confusion Matrix:")
    logger.info("  %s", results['confusion_matrix'][0])
    logger.info("  %s", results['confusion_matrix'][1])
    
    # Save results
    logger.info("="*60)
    logger.info("Saving results to %s...", output_dir)
    
    # Generate plots
    prefix = f"payload_{MIN_PAYLOAD or 'min'}_{MAX_PAYLOAD or 'max'}"
    plot_results(results, output_dir, prefix=prefix)
    
    # Save detailed results to text file
    with open(output_dir / f'{prefix}_results.txt', 'w') as f:
        f.write("PAYLOAD-FILTERED MODEL TESTING RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {excel_path}\n")
        f.write(f"Payload filter: min={MIN_PAYLOAD}, max={MAX_PAYLOAD}\n\n")
        f.write("Dataset Statistics:\n")
        f.write(f"  Total images: {stats['total_images']}\n")
        f.write(f"  Total clean images: {stats['total_clean']}\n")
        f.write(f"  Total stego images: {stats['total_stego']}\n")
        f.write(f"  Filtered images: {stats['filtered_count']} (Clean: {stats['filtered_clean']}, Stego: {stats['filtered_stego']})\n")
        f.write(f"  Stego payload range: [{stats['min_payload']:.6f}, {stats['max_payload']:.6f}]\n")
        f.write(f"  Mean payload: {stats['mean_payload']:.6f}\n")
        f.write(f"  Median payload: {stats['median_payload']:.6f}\n")
        f.write(f"  Std payload: {stats['std_payload']:.6f}\n\n")
        f.write("Evaluation Results:\n")
        f.write(f"  Loss: {results['loss']:.4f}\n")
        f.write(f"  Accuracy: {results['accuracy']:.2f}%\n")
        if results['auc'] is not None:
            f.write(f"  AUC: {results['auc']:.4f}\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  {results['confusion_matrix'][0]}\n")
        f.write(f"  {results['confusion_matrix'][1]}\n")
    
    logger.info("Results saved successfully!")
    logger.info("  - Confusion matrix: %s", output_dir / f'{prefix}_confusion_matrix.png')
    if results['auc'] is not None:
        logger.info("  - ROC curve: %s", output_dir / f'{prefix}_roc_curve.png')
    logger.info("  - Detailed results: %s", output_dir / f'{prefix}_results.txt')
    logger.info("="*60)


if __name__ == '__main__':
    main()
