#!/usr/bin/env python3
"""
JPEG Quality Factor Detection for BOSS Dataset

This script analyzes JPEG quantization tables to estimate the quality factor (QF)
used during compression. Helps determine if BOSS images are QF 75 or QF 95.

The quality factor estimation is based on comparing the image's quantization table
with standard JPEG quantization tables at different quality settings.

Author: Kaleb Austgen
Date: November 3, 2025
"""

import numpy as np
from PIL import Image
from pathlib import Path
import logging
from collections import Counter
from typing import Dict, List, Tuple

# ============================================================
# CONFIGURATION
# ============================================================

# Path to BOSS dataset directory
BOSS_DIR = Path(__file__).parent.parent / 'BOSS1.01_Dataset'

# Number of images to analyze (sample size)
NUM_SAMPLES = 1000

# Enable verbose output (detailed analysis for each image)
VERBOSE = False

# ============================================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Standard JPEG luminance quantization table (quality 50)
STANDARD_LUMINANCE_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def scale_quantization_table(quality: int) -> np.ndarray:
    """
    Scale the standard quantization table for a given quality factor.
    
    JPEG quality scaling formula (from IJG libjpeg):
    - quality < 50: scale = 5000 / quality
    - quality >= 50: scale = 200 - 2 * quality
    
    Args:
        quality: JPEG quality factor (1-100)
    
    Returns:
        Scaled 8x8 quantization table
    """
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    # Scale the standard table
    scaled_qt = np.floor((STANDARD_LUMINANCE_QT * scale + 50) / 100)
    
    # Clip values to valid range [1, 255]
    scaled_qt = np.clip(scaled_qt, 1, 255)
    
    return scaled_qt.astype(np.uint8)


def extract_quantization_table(image_path: Path) -> np.ndarray:
    """
    Extract the luminance quantization table from a JPEG image.
    
    Args:
        image_path: Path to JPEG image
    
    Returns:
        8x8 quantization table as numpy array, or None if extraction fails
    """
    try:
        img = Image.open(image_path)
        
        # Check if it's a JPEG
        if img.format != 'JPEG':
            logger.warning(f"{image_path.name} is not a JPEG (format: {img.format})")
            return None
        
        # Get quantization tables from JPEG info
        if hasattr(img, 'quantization'):
            # PIL stores quantization tables as a dict
            qtables = img.quantization
            
            if qtables and len(qtables) > 0:
                # Get the luminance (Y) quantization table (key 0)
                qt_flat = qtables[0]  # Returns flat list
                
                # Reshape to 8x8
                qt = np.array(qt_flat).reshape(8, 8)
                return qt
        
        logger.warning(f"Could not extract quantization table from {image_path.name}")
        return None
        
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return None


def estimate_quality_factor(qt: np.ndarray) -> Tuple[int, float]:
    """
    Estimate the quality factor by comparing with standard tables.
    
    Uses sum of squared differences (SSD) to find best match.
    
    Args:
        qt: 8x8 quantization table from image
    
    Returns:
        (estimated_quality, confidence_score)
    """
    best_quality = None
    min_ssd = float('inf')
    
    # Test quality factors from 50 to 100 (most common range)
    for quality in range(50, 101):
        reference_qt = scale_quantization_table(quality)
        ssd = np.sum((qt - reference_qt) ** 2)
        
        if ssd < min_ssd:
            min_ssd = ssd
            best_quality = quality
    
    # Also check lower qualities (40-49) in case of unusual compression
    for quality in range(40, 50):
        reference_qt = scale_quantization_table(quality)
        ssd = np.sum((qt - reference_qt) ** 2)
        
        if ssd < min_ssd:
            min_ssd = ssd
            best_quality = quality
    
    # Calculate confidence (lower SSD = higher confidence)
    # Perfect match has SSD = 0, confidence = 1.0
    # Higher SSD reduces confidence
    confidence = 1.0 / (1.0 + min_ssd / 100.0)
    
    return best_quality, confidence


def analyze_boss_dataset(boss_dir: Path, num_samples: int = 100) -> Dict:
    """
    Analyze a sample of BOSS images to determine their quality factor.
    
    Args:
        boss_dir: Path to BOSS dataset directory
        num_samples: Number of images to analyze
    
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing BOSS dataset: {boss_dir}")
    logger.info(f"Sampling {num_samples} images...")
    
    # Get all JPEG images
    image_files = sorted(list(boss_dir.glob("*.jpg")) + list(boss_dir.glob("*.jpeg")))
    
    if not image_files:
        logger.error("No JPEG images found in directory!")
        return None
    
    logger.info(f"Found {len(image_files)} JPEG images")
    
    # Sample uniformly across the dataset
    step = max(1, len(image_files) // num_samples)
    sampled_files = image_files[::step][:num_samples]
    
    logger.info(f"Analyzing {len(sampled_files)} images...")
    
    quality_estimates = []
    confidence_scores = []
    quantization_tables = []
    
    for img_path in sampled_files:
        qt = extract_quantization_table(img_path)
        
        if qt is not None:
            quality, confidence = estimate_quality_factor(qt)
            quality_estimates.append(quality)
            confidence_scores.append(confidence)
            quantization_tables.append(qt)
    
    if not quality_estimates:
        logger.error("Failed to analyze any images!")
        return None
    
    # Statistical analysis
    quality_counts = Counter(quality_estimates)
    most_common_quality = quality_counts.most_common(1)[0][0]
    
    results = {
        'num_analyzed': len(quality_estimates),
        'quality_estimates': quality_estimates,
        'confidence_scores': confidence_scores,
        'mean_quality': np.mean(quality_estimates),
        'median_quality': np.median(quality_estimates),
        'std_quality': np.std(quality_estimates),
        'most_common_quality': most_common_quality,
        'quality_distribution': dict(quality_counts),
        'mean_confidence': np.mean(confidence_scores),
        'sample_quantization_table': quantization_tables[0] if quantization_tables else None
    }
    
    return results


def print_results(results: Dict):
    """Print analysis results in a readable format."""
    
    print("\n" + "="*70)
    print("BOSS DATASET QUALITY FACTOR ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Sample Size: {results['num_analyzed']} images")
    
    print(f"\n🎯 Quality Factor Estimates:")
    print(f"   Most Common QF: {results['most_common_quality']}")
    print(f"   Mean QF: {results['mean_quality']:.2f}")
    print(f"   Median QF: {results['median_quality']:.1f}")
    print(f"   Std Dev: {results['std_quality']:.2f}")
    
    print(f"\n✅ Detection Confidence: {results['mean_confidence']:.1%}")
    
    print(f"\n📈 Quality Factor Distribution:")
    sorted_dist = sorted(results['quality_distribution'].items())
    for qf, count in sorted_dist:
        percentage = (count / results['num_analyzed']) * 100
        bar = "█" * int(percentage / 2)
        print(f"   QF {qf:3d}: {count:3d} images ({percentage:5.1f}%) {bar}")
    
    # Conclusion
    most_common = results['most_common_quality']
    print(f"\n🔍 Conclusion:")
    
    if results['std_quality'] < 2.0:
        print(f"   ✅ Dataset is UNIFORMLY compressed at QF {most_common}")
        
    else:
        print(f"   ⚠️  Dataset has MIXED quality factors (std={results['std_quality']:.2f})")
        print(f"   📌 Most images are QF {most_common}, but significant variation exists")
    
    # Show sample quantization table
    if results['sample_quantization_table'] is not None:
        print(f"\n📋 Sample Quantization Table (first image):")
        qt = results['sample_quantization_table']
        print("   Luminance (Y) channel:")
        for row in qt:
            print("   ", " ".join(f"{val:3d}" for val in row))
        
        # Compare with known standards
        print(f"\n   Reference QF 75 table (first row):")
        qf75_table = scale_quantization_table(75)
        print("   ", " ".join(f"{val:3d}" for val in qf75_table[0]))
        
        print(f"\n   Reference QF 95 table (first row):")
        qf95_table = scale_quantization_table(95)
        print("   ", " ".join(f"{val:3d}" for val in qf95_table[0]))
        
        print(f"\n   Actual image table (first row):")
        print("   ", " ".join(f"{val:3d}" for val in qt[0]))
    
    print("\n" + "="*70)


def main():
    """Main function - uses global configuration variables."""
    
    if VERBOSE:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if directory exists
    if not BOSS_DIR.exists():
        logger.error(f"Directory not found: {BOSS_DIR}")
        logger.info("Please edit BOSS_DIR at the top of this script")
        return 1
    
    # Run analysis
    results = analyze_boss_dataset(BOSS_DIR, num_samples=NUM_SAMPLES)
    
    if results is None:
        logger.error("Analysis failed!")
        return 1
    
    # Print results
    print_results(results)
    
    return 0


if __name__ == "__main__":
    exit(main())
