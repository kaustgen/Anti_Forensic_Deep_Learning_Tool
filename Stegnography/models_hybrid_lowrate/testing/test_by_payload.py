#!/usr/bin/env python3
"""
Test Hybrid Spatial+Frequency CNN by payload rate.

Evaluates the trained hybrid model on different J-UNIWARD embedding rates
(0.1, 0.2, 0.3, 0.4 bpnzAC) and generates comprehensive evaluation metrics:
    - Confusion matrices per payload rate
    - ROC curves with AUC scores
    - Precision-Recall curves
    - Per-class metrics (precision, recall, F1-score)
    - Overall accuracy across all rates

Usage:
    python test_by_payload.py

Configuration:
    - MODEL_PATH: Path to trained model checkpoint (best_hybrid_model.pth)
    - EXCEL_PATH: Path to dataset Excel with payload information
    - OUTPUT_DIR: Directory to save test results and plots
    - BATCH_SIZE: Batch size for evaluation (larger = faster)
    - DEVICE: 'cuda', 'cpu', or None for auto-detect

Author: Kaleb Austgen
Date: October 30, 2025
"""

import sys
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent.parent))
from sten_dct_cnn import SpatialFrequencyHybridCNN, HybridSpatialDCTDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
MODEL_PATH = Path(__file__).parent.parent / 'grey_only/best_hybrid_model.pth'
EXCEL_PATH = Path(__file__).parent.parent.parent.parent / 'dataGen' / 'RGB_stego_metadata.xlsx'
IMG_ROOT = Path(__file__).parent.parent.parent.parent / 'dataGen'
CACHE_DIR = Path(__file__).parent.parent.parent / '.dct_rgb_cache'
OUTPUT_DIR = Path(__file__).parent / 'results'

# Evaluation parameters
BATCH_SIZE = 64  # Larger batch size for faster evaluation (no gradients needed)
DEVICE = None    # None for auto-detect, 'cuda', or 'cpu'

# Payload rates to test (J-UNIWARD bpnzAC values)
PAYLOAD_RATES = [0.1, 0.2, 0.3, 0.4]

# Payload tolerance (±) for filtering images
# e.g., 0.1 ± 0.02 means images with payload 0.08-0.12
PAYLOAD_TOLERANCE = 0.04


# ============================================================
# DATASET FILTERING
# ============================================================

def filter_by_payload(excel_path, target_payload, tolerance=0.02, payload_column='Payload (bpp AC DCT)'):
    """
    Filter dataset indices by payload rate.
    
    Returns balanced dataset:
        - Stego images with payload in range [target - tolerance, target + tolerance]
        - Equal number of randomly sampled clean images
    
    Args:
        excel_path: Path to Excel with payload statistics
        target_payload: Target payload rate (e.g., 0.1, 0.2, 0.3, 0.4)
        tolerance: Allowed deviation from target (e.g., ±0.02)
        payload_column: Column name containing payload values
    
    Returns:
        tuple: (filtered_indices, stats_dict)
    """
    df = pd.read_excel(excel_path)
    
    if payload_column not in df.columns:
        raise ValueError(f"Column '{payload_column}' not found. Available: {list(df.columns)}")
    
    # Identify stego vs clean images
    stego_mask = df.iloc[:, 1] == True  # Column 2: "Stegnography Applied?"
    
    # Filter stego images by payload range
    stego_indices = []
    payload_values = []
    
    min_payload = target_payload - tolerance
    max_payload = target_payload + tolerance
    
    for idx in range(len(df)):
        if not stego_mask.iloc[idx]:
            continue  # Skip clean images
        
        payload_val = df[payload_column].iloc[idx]
        
        # Handle missing/invalid values
        if pd.isna(payload_val) or payload_val in ['N/A', 'ERROR', 'NO COVER']:
            continue
        
        try:
            payload_float = float(payload_val)
        except (ValueError, TypeError):
            continue
        
        # Check if payload is in target range
        if min_payload <= payload_float <= max_payload:
            stego_indices.append(idx)
            payload_values.append(payload_float)
    
    # Get all clean image indices
    clean_indices = [idx for idx in range(len(df)) if not stego_mask.iloc[idx]]
    
    # Randomly sample clean images to match stego count
    num_stego = len(stego_indices)
    num_clean_available = len(clean_indices)
    
    if num_clean_available > 0:
        num_clean = min(num_stego, num_clean_available)
        np.random.seed(42)  # Reproducible sampling
        sampled_clean_indices = np.random.choice(clean_indices, size=num_clean, replace=False).tolist()
    else:
        sampled_clean_indices = []
        num_clean = 0
    
    # Combine stego + clean
    filtered_indices = stego_indices + sampled_clean_indices
    
    # Statistics
    stats = {
        'target_payload': target_payload,
        'tolerance': tolerance,
        'min_payload': min_payload,
        'max_payload': max_payload,
        'total_filtered': len(filtered_indices),
        'num_clean': num_clean,
        'num_stego': num_stego,
        'actual_min_payload': np.min(payload_values) if payload_values else None,
        'actual_max_payload': np.max(payload_values) if payload_values else None,
        'mean_payload': np.mean(payload_values) if payload_values else None,
        'std_payload': np.std(payload_values) if payload_values else None,
    }
    
    return filtered_indices, stats


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_subset(model, dataset, indices, device, batch_size):
    """
    Evaluate model on subset of dataset.
    
    Args:
        model: Trained SpatialFrequencyHybridCNN
        dataset: Full HybridSpatialDCTDataset
        indices: List of indices to evaluate
        device: 'cuda' or 'cpu'
        batch_size: Batch size for evaluation
    
    Returns:
        dict: Evaluation results (loss, accuracy, predictions, labels, etc.)
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probs = []  # Probability of stego class
    
    with torch.no_grad():
        for spatial_img, dct_blocks, labels in tqdm(loader, desc="Evaluating", leave=False):
            spatial_img = spatial_img.to(device)
            dct_blocks = dct_blocks.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(spatial_img, dct_blocks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Get probability of stego class (class 1)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    
    # AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = None
    
    # Average precision
    try:
        avg_precision = average_precision_score(all_labels, all_probs)
    except:
        avg_precision = None
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probs),
        'confusion_matrix': cm,
        'auc': auc,
        'average_precision': avg_precision,
    }


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_confusion_matrix(cm, title, output_path):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Clean', 'Stego'],
        yticklabels=['Clean', 'Stego'],
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_roc_curve(results_dict, output_path):
    """Plot ROC curves for all payload rates."""
    plt.figure(figsize=(10, 8))
    
    for payload_rate, results in results_dict.items():
        if results['auc'] is None:
            continue
        
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        plt.plot(
            fpr, tpr, 
            label=f"{payload_rate} bpnzAC (AUC = {results['auc']:.3f})",
            linewidth=2
        )
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves by Payload Rate', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_precision_recall_curve(results_dict, output_path):
    """Plot Precision-Recall curves for all payload rates."""
    plt.figure(figsize=(10, 8))
    
    for payload_rate, results in results_dict.items():
        if results['average_precision'] is None:
            continue
        
        precision, recall, _ = precision_recall_curve(
            results['labels'], 
            results['probabilities']
        )
        
        plt.plot(
            recall, precision,
            label=f"{payload_rate} bpnzAC (AP = {results['average_precision']:.3f})",
            linewidth=2
        )
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves by Payload Rate', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_accuracy_by_payload(results_dict, output_path):
    """Plot accuracy vs payload rate."""
    payload_rates = sorted(results_dict.keys())
    accuracies = [results_dict[rate]['accuracy'] for rate in payload_rates]
    
    plt.figure(figsize=(10, 6))
    plt.plot(payload_rates, accuracies, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    
    # Add value labels
    for rate, acc in zip(payload_rates, accuracies):
        plt.text(rate, acc + 1, f'{acc:.1f}%', ha='center', fontsize=10)
    
    plt.xlabel('Payload Rate (bpnzAC)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Detection Accuracy by Payload Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_combined_confusion_matrix(results_dict, output_path):
    """Plot combined confusion matrix across all payload rates."""
    # Combine all predictions and labels
    all_labels = []
    all_predictions = []
    
    for results in results_dict.values():
        all_labels.extend(results['labels'])
        all_predictions.extend(results['predictions'])
    
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Clean', 'Stego'],
        yticklabels=['Clean', 'Stego'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Combined Confusion Matrix (All Payload Rates)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path}")


# ============================================================
# MAIN TESTING PIPELINE
# ============================================================

def main():
    """Run comprehensive testing by payload rate."""
    
    # Setup device
    if DEVICE is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = DEVICE
    
    logger.info("="*70)
    logger.info("HYBRID SPATIAL+FREQUENCY CNN - PAYLOAD-BASED TESTING")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Dataset: {EXCEL_PATH}")
    logger.info(f"Payload rates: {PAYLOAD_RATES}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info("="*70)
    
    # Check files exist
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return
    
    if not EXCEL_PATH.exists():
        logger.error(f"Dataset Excel not found: {EXCEL_PATH}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========== LOAD MODEL ==========
    logger.info("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Extract model configuration
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_acc', 0.0)
        logger.info(f"  Checkpoint from epoch {epoch} (val_acc: {val_acc:.2f}%)")
    else:
        state_dict = checkpoint
        logger.info("  Loading direct state_dict")
    
    # Create model
    model = SpatialFrequencyHybridCNN(
        num_channels=3,
        target_blocks=(32, 32),
        dropout=0.3  # Match training config
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # ========== LOAD DATASET ==========
    logger.info("\nLoading dataset...")
    dataset = HybridSpatialDCTDataset(
        excel_path=EXCEL_PATH,
        img_root=IMG_ROOT,
        dct_channels=['Y', 'Cb', 'Cr'],
        target_blocks=(32, 32),
        cache_dir=CACHE_DIR,
        train=False  # No augmentation for testing
    )
    logger.info(f"  Total dataset size: {len(dataset)}")
    
    # ========== TEST EACH PAYLOAD RATE ==========
    results_by_payload = {}
    
    for payload_rate in PAYLOAD_RATES:
        logger.info("\n" + "="*70)
        logger.info(f"TESTING PAYLOAD RATE: {payload_rate} bpnzAC")
        logger.info("="*70)
        
        # Filter dataset
        logger.info(f"Filtering images with payload {payload_rate} ± {PAYLOAD_TOLERANCE}...")
        indices, stats = filter_by_payload(
            EXCEL_PATH,
            target_payload=payload_rate,
            tolerance=PAYLOAD_TOLERANCE
        )
        
        logger.info(f"  Filtered images: {stats['total_filtered']}")
        logger.info(f"    Clean: {stats['num_clean']}")
        logger.info(f"    Stego: {stats['num_stego']}")
        
        if stats['actual_min_payload'] is not None and stats['actual_max_payload'] is not None:
            logger.info(f"  Actual payload range: [{stats['actual_min_payload']:.4f}, {stats['actual_max_payload']:.4f}]")
            logger.info(f"  Mean payload: {stats['mean_payload']:.4f}")
        else:
            logger.info(f"  Actual payload range: No stego images found")
            logger.info(f"  Mean payload: N/A")
        
        if stats['total_filtered'] == 0:
            logger.warning(f"No images found for payload {payload_rate}!")
            continue
        
        # Evaluate
        logger.info(f"Evaluating model on {stats['total_filtered']} images...")
        results = evaluate_subset(model, dataset, indices, device, BATCH_SIZE)
        results['stats'] = stats
        results_by_payload[payload_rate] = results
        
        # Log results
        logger.info(f"\nResults for {payload_rate} bpnzAC:")
        logger.info(f"  Loss: {results['loss']:.4f}")
        logger.info(f"  Accuracy: {results['accuracy']:.2f}%")
        if results['auc'] is not None:
            logger.info(f"  AUC: {results['auc']:.4f}")
        if results['average_precision'] is not None:
            logger.info(f"  Average Precision: {results['average_precision']:.4f}")
        
        # Confusion matrix
        cm = results['confusion_matrix']
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Classification report
        report = classification_report(
            results['labels'],
            results['predictions'],
            labels=[0, 1],
            target_names=['Clean', 'Stego'],
            zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix for this payload rate
        plot_confusion_matrix(
            cm,
            title=f'Confusion Matrix - {payload_rate} bpnzAC',
            output_path=OUTPUT_DIR / f'confusion_matrix_{payload_rate}.png'
        )
    
    # ========== GENERATE COMBINED PLOTS ==========
    logger.info("\n" + "="*70)
    logger.info("GENERATING SUMMARY PLOTS")
    logger.info("="*70)
    
    if results_by_payload:
        # ROC curves
        plot_roc_curve(
            results_by_payload,
            output_path=OUTPUT_DIR / 'roc_curves_all_payloads.png'
        )
        
        # Precision-Recall curves
        plot_precision_recall_curve(
            results_by_payload,
            output_path=OUTPUT_DIR / 'precision_recall_curves.png'
        )
        
        # Accuracy vs payload
        plot_accuracy_by_payload(
            results_by_payload,
            output_path=OUTPUT_DIR / 'accuracy_vs_payload.png'
        )
        
        # Combined confusion matrix
        plot_combined_confusion_matrix(
            results_by_payload,
            output_path=OUTPUT_DIR / 'confusion_matrix_combined.png'
        )
    
    # ========== SAVE DETAILED RESULTS ==========
    logger.info("\n" + "="*70)
    logger.info("SAVING DETAILED RESULTS")
    logger.info("="*70)
    
    with open(OUTPUT_DIR / 'test_results.txt', 'w') as f:
        f.write("HYBRID SPATIAL+FREQUENCY CNN - PAYLOAD-BASED TESTING\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Dataset: {EXCEL_PATH}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Payload tolerance: ±{PAYLOAD_TOLERANCE}\n\n")
        
        for payload_rate, results in sorted(results_by_payload.items()):
            f.write("="*70 + "\n")
            f.write(f"PAYLOAD RATE: {payload_rate} bpnzAC\n")
            f.write("="*70 + "\n\n")
            
            stats = results['stats']
            f.write("Dataset Statistics:\n")
            f.write(f"  Total filtered: {stats['total_filtered']}\n")
            f.write(f"  Clean images: {stats['num_clean']}\n")
            f.write(f"  Stego images: {stats['num_stego']}\n")
            
            if stats['actual_min_payload'] is not None and stats['actual_max_payload'] is not None:
                f.write(f"  Payload range: [{stats['actual_min_payload']:.4f}, {stats['actual_max_payload']:.4f}]\n")
                f.write(f"  Mean payload: {stats['mean_payload']:.4f}\n\n")
            else:
                f.write(f"  Payload range: No stego images found\n")
                f.write(f"  Mean payload: N/A\n\n")
            
            f.write("Evaluation Results:\n")
            f.write(f"  Loss: {results['loss']:.4f}\n")
            f.write(f"  Accuracy: {results['accuracy']:.2f}%\n")
            if results['auc'] is not None:
                f.write(f"  AUC: {results['auc']:.4f}\n")
            if results['average_precision'] is not None:
                f.write(f"  Average Precision: {results['average_precision']:.4f}\n")
            
            cm = results['confusion_matrix']
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],\n")
            f.write(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]\n\n")
            
            report = classification_report(
                results['labels'],
                results['predictions'],
                labels=[0, 1],
                target_names=['Clean', 'Stego'],
                zero_division=0
            )
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\n")
        
        # Summary table
        f.write("="*70 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Payload':<12} {'Accuracy':<12} {'AUC':<12} {'Avg Precision':<15}\n")
        f.write("-"*70 + "\n")
        for payload_rate in sorted(results_by_payload.keys()):
            results = results_by_payload[payload_rate]
            auc_str = f"{results['auc']:.4f}" if results['auc'] is not None else "N/A"
            ap_str = f"{results['average_precision']:.4f}" if results['average_precision'] is not None else "N/A"
            f.write(f"{payload_rate:<12.1f} {results['accuracy']:<12.2f} {auc_str:<12} {ap_str:<15}\n")
    
    logger.info(f"  Saved: {OUTPUT_DIR / 'test_results.txt'}")
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*70)
    logger.info("TESTING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    logger.info("  - test_results.txt (detailed results)")
    logger.info("  - confusion_matrix_*.png (per-payload confusion matrices)")
    logger.info("  - confusion_matrix_combined.png (all payloads)")
    logger.info("  - roc_curves_all_payloads.png")
    logger.info("  - precision_recall_curves.png")
    logger.info("  - accuracy_vs_payload.png")
    logger.info("="*70)


if __name__ == '__main__':
    main()
