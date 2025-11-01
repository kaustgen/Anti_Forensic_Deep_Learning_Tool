#!/usr/bin/env python3
"""
Combine BOSS and RGB datasets into unified training dataset.

Creates a single Excel file that merges:
- BOSS dataset: 20k images (10k covers + 10k stego) - grayscale nature scenes
- RGB dataset: ~4k images (~2k covers + ~2k stego) - diverse color images

The combined dataset provides:
1. Strong baseline from BOSS (proven steganalysis dataset)
2. Real-world diversity from RGB (various content, color channels)
3. Better generalization to real-world steganography detection

Author: Kaleb Austgen
Date: October 30, 2025
"""

import logging
from pathlib import Path
from openpyxl import load_workbook, Workbook
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

# Input datasets
BOSS_EXCEL = BASE_DIR / 'BOSS_stego_metadata.xlsx'
RGB_EXCEL = BASE_DIR / 'RGB_stego_metadata.xlsx'

# Output combined dataset
COMBINED_EXCEL = BASE_DIR / 'BOSS_RGB_combined.xlsx'

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_dataset(df, name):
    """
    Validate dataset structure and contents.
    
    Both BOSS and RGB datasets now share identical column structure:
        A: File Path
        B: Stegnography Applied?
        C: Payload Category
        D: Payload Size (bytes)
        E: Payload (bpp AC DCT)
        F: Payload (bytes)
        G: Payload (bits)
        H: Image Dimensions
        I: Non-zero AC DCT
    
    Args:
        df: DataFrame to validate
        name: Dataset name for logging
        
    Returns:
        bool: True if valid, False otherwise
    """
    logger.info(f"\nValidating {name} dataset...")
    
    if df.empty:
        logger.error(f"{name} dataset is empty!")
        return False
    
    # Expected column structure (standardized across both datasets)
    expected_columns = [
        'File Path',
        'Stegnography Applied?',
        'Payload Category',
        'Payload Size (bytes)',
        'Payload (bpp AC DCT)',
        'Payload (bytes)',
        'Payload (bits)',
        'Image Dimensions',
        'Non-zero AC DCT',
        'RGB'  # New column to track RGB vs grayscale
    ]
    
    # Check if all expected columns exist
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"{name} missing required columns: {missing_cols}")
        logger.error(f"Found columns: {list(df.columns)}")
        return False
    
    # Verify column order matches
    actual_cols = list(df.columns)[:len(expected_columns)]
    if actual_cols != expected_columns:
        logger.warning(f"{name} column order differs from expected")
        logger.warning(f"Expected: {expected_columns}")
        logger.warning(f"Actual:   {actual_cols}")
    
    # Count entries
    total_rows = len(df)
    logger.info(f"{name}: {total_rows} entries")
    
    # Count clean vs stego using standardized column
    stego_col = 'Stegnography Applied?'
    try:
        if df[stego_col].dtype == bool:
            stego_count = df[stego_col].sum()
        else:
            # Handle string 'True'/'False' values
            stego_count = (df[stego_col] == True).sum() + (df[stego_col] == 'True').sum()
        
        clean_count = total_rows - stego_count
        logger.info(f"    - Clean images: {clean_count}")
        logger.info(f"    - Stego images: {stego_count}")
        
        # Validate 1:1 ratio
        if clean_count != stego_count:
            logger.warning(f"    ⚠️  Unbalanced dataset: {clean_count} clean vs {stego_count} stego")
        
    except Exception as e:
        logger.error(f"Failed to count stego images: {e}")
        return False
    
    logger.info(f"✓ {name} dataset structure validated")
    return True


def get_dataset_stats(df, name):
    """
    Get detailed statistics about dataset.
    
    Uses standardized column structure (identical for BOSS and RGB).
    
    Args:
        df: DataFrame
        name: Dataset name
        
    Returns:
        dict: Statistics
    """
    stats = {
        'name': name,
        'total': len(df),
        'clean': 0,
        'stego': 0,
        'by_category': {}
    }
    
    # Use standardized column names
    stego_col = 'Stegnography Applied?'
    category_col = 'Payload Category'
    
    try:
        # Count clean vs stego
        if df[stego_col].dtype == bool:
            stats['stego'] = df[stego_col].sum()
        else:
            stats['stego'] = (df[stego_col] == True).sum() + (df[stego_col] == 'True').sum()
        
        stats['clean'] = stats['total'] - stats['stego']
        
        # Count by payload category
        if category_col in df.columns:
            category_counts = df[category_col].value_counts().to_dict()
            # Remove N/A (clean images)
            stats['by_category'] = {k: v for k, v in category_counts.items() if k != 'N/A'}
    
    except Exception as e:
        logger.warning(f"Failed to calculate detailed stats for {name}: {e}")
    
    return stats


# ============================================================
# MAIN COMBINATION PIPELINE
# ============================================================

def main():
    """Combine BOSS and RGB datasets into unified training set."""
    
    logger.info("="*70)
    logger.info("BOSS + RGB DATASET COMBINATION")
    logger.info("="*70)
    
    # Check input files exist
    if not BOSS_EXCEL.exists():
        logger.error(f"BOSS dataset not found: {BOSS_EXCEL}")
        logger.error("Run steg_mass_gen_j_uniward.py first!")
        return
    
    if not RGB_EXCEL.exists():
        logger.error(f"RGB dataset not found: {RGB_EXCEL}")
        logger.error("Run steg_rgb_j_uniward.py first!")
        return
    
    logger.info(f"BOSS dataset:         {BOSS_EXCEL}")
    logger.info(f"RGB dataset:          {RGB_EXCEL}")
    logger.info(f"Combined output:      {COMBINED_EXCEL}")
    logger.info("="*70)
    
    # Load datasets
    logger.info("\nLoading datasets...")
    try:
        boss_df = pd.read_excel(BOSS_EXCEL)
        logger.info(f"Loaded BOSS: {len(boss_df)} entries")
    except Exception as e:
        logger.error(f"Failed to load BOSS dataset: {e}")
        return
    
    try:
        rgb_df = pd.read_excel(RGB_EXCEL)
        logger.info(f"Loaded RGB: {len(rgb_df)} entries")
    except Exception as e:
        logger.error(f"Failed to load RGB dataset: {e}")
        return
    
    # Validate datasets
    if not validate_dataset(boss_df, "BOSS"):
        return
    
    if not validate_dataset(rgb_df, "RGB"):
        return
    
    # Get statistics
    boss_stats = get_dataset_stats(boss_df, "BOSS")
    rgb_stats = get_dataset_stats(rgb_df, "RGB")
    
    # Combine datasets
    logger.info("\nCombining datasets...")
    logger.info("Both datasets now use identical column structure (standardized)")
    
    # Since both datasets have identical columns, simple concatenation works
    try:
        combined_df = pd.concat([boss_df, rgb_df], ignore_index=True)
        logger.info(f"✓ Successfully combined {len(boss_df)} BOSS + {len(rgb_df)} RGB = {len(combined_df)} total rows")
    except Exception as e:
        logger.error(f"Failed to combine datasets: {e}")
        return
    
    combined_stats = {
        'total': len(combined_df),
        'boss': len(boss_df),
        'rgb': len(rgb_df)
    }
    
    # Calculate RGB distribution
    rgb_distribution = {
        'boss_grayscale': (boss_df['RGB'] == False).sum(),
        'boss_rgb': (boss_df['RGB'] == True).sum(),
        'rgb_dataset': (rgb_df['RGB'] == True).sum(),
        'total_grayscale': (combined_df['RGB'] == False).sum(),
        'total_rgb': (combined_df['RGB'] == True).sum()
    }
    
    # Save combined dataset
    logger.info("\nSaving combined dataset...")
    try:
        combined_df.to_excel(COMBINED_EXCEL, index=False, engine='openpyxl')
        logger.info(f"Combined dataset saved: {COMBINED_EXCEL}")
    except Exception as e:
        logger.error(f"Failed to save combined dataset: {e}")
        return
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("COMBINATION COMPLETE")
    logger.info("="*70)
    logger.info("\nDataset Breakdown:")
    logger.info(f"  BOSS dataset:       {boss_stats['total']:,} entries")
    logger.info(f"    - Clean:          {boss_stats['clean']:,}")
    logger.info(f"    - Stego:          {boss_stats['stego']:,}")
    if boss_stats['by_category']:
        logger.info(f"    - By rate:        {', '.join(f'{k}={v}' for k, v in boss_stats['by_category'].items())}")
    logger.info("")
    logger.info(f"  RGB dataset:        {rgb_stats['total']:,} entries")
    logger.info(f"    - Clean:          {rgb_stats['clean']:,}")
    logger.info(f"    - Stego:          {rgb_stats['stego']:,}")
    if rgb_stats['by_category']:
        logger.info(f"    - By rate:        {', '.join(f'{k}={v}' for k, v in rgb_stats['by_category'].items())}")
    logger.info("")
    logger.info(f"  Combined total:     {combined_stats['total']:,} entries")
    logger.info(f"    - BOSS contrib:   {combined_stats['boss']:,} ({100*combined_stats['boss']/combined_stats['total']:.1f}%)")
    logger.info(f"    - RGB contrib:    {combined_stats['rgb']:,} ({100*combined_stats['rgb']/combined_stats['total']:.1f}%)")
    logger.info("")
    logger.info(f"Column structure:     9 columns (standardized)")
    logger.info(f"  1. File Path")
    logger.info(f"  2. Stegnography Applied?")
    logger.info(f"  3. Payload Category")
    logger.info(f"  4. Payload Size (bytes)")
    logger.info(f"  5. Payload (bpp AC DCT)")
    logger.info(f"  6. Payload (bytes)")
    logger.info(f"  7. Payload (bits)")
    logger.info(f"  8. Image Dimensions")
    logger.info(f"  9. Non-zero AC DCT")
    logger.info(f" 10. RGB")
    logger.info("")
    logger.info(f"RGB Distribution:")
    logger.info(f"  Grayscale (BOSS): {rgb_distribution['total_grayscale']:,} ({100*rgb_distribution['total_grayscale']/combined_stats['total']:.1f}%)")
    logger.info(f"  Color (RGB):      {rgb_distribution['total_rgb']:,} ({100*rgb_distribution['total_rgb']/combined_stats['total']:.1f}%)")
    logger.info("")
    logger.info(f"Output file:          {COMBINED_EXCEL}")
    logger.info("="*70)

if __name__ == '__main__':
    main()
