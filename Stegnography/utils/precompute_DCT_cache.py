# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose: Run this before train_hybrid.py to precompute all spatial DCT features
#          Uses multiprocessing for speed (10x faster than sequential)

import sys
from pathlib import Path

# Add parent directory to path to import sten_dct_cnn
sys.path.insert(0, str(Path(__file__).parent.parent))

from sten_dct_cnn import HybridSpatialDCTDataset
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Excel path - use combined BOSS+RGB dataset
EXCEL_PATH = Path(__file__).parent.parent.parent / 'dataGen' / 'RGB_stego_metadata.xlsx'

# Image root directory (not used if paths in Excel are absolute)
IMG_ROOT = Path('.')

# DCT channels to cache
DCT_CHANNELS = ['Y', 'Cb', 'Cr']

# Target block size (32×32 blocks = 256×256 pixels)
TARGET_BLOCKS = (32, 32)

# Cache directory (will be created if doesn't exist)
CACHE_DIR = Path(__file__).parent.parent / '.dct_BOSS_RGB_combined_cache'

# Number of worker processes (leave 1 CPU free for system)
NUM_WORKERS = max(1, cpu_count() - 1)


def process_image(args):
    """
    Worker function to cache a single image's DCT features.
    
    This runs in a separate process for parallel processing.
    Must reimport dependencies inside worker for multiprocessing compatibility.
    """
    dataset_config, idx = args
    
    # Import inside worker (required for multiprocessing)
    import torch
    import numpy as np
    import jpegio as jio
    import pandas as pd
    from hashlib import sha1
    from pathlib import Path
    
    # Reconstruct dataset configuration
    excel_path = dataset_config['excel_path']
    img_root = Path(dataset_config['img_root'])
    dct_channels = dataset_config['dct_channels']
    target_blocks = tuple(dataset_config['target_blocks'])
    cache_dir = Path(dataset_config['cache_dir'])
    path_col = dataset_config['path_col']
    
    # Load this specific row from Excel
    try:
        records = pd.read_excel(excel_path, header=0)
        row = records.iloc[idx]
        img_path = Path(row[path_col])
        
        # Make absolute path if needed
        if not img_path.is_absolute():
            img_path = (img_root / img_path).resolve()
        
        # Check if file exists
        if not img_path.exists():
            return f'error: File not found: {img_path}'
        
    except Exception as e:
        return f'error: Failed to read Excel row {idx}: {e}'
    
    # Generate cache filename from image path hash
    h = sha1(str(img_path).encode('utf-8')).hexdigest()
    cache_file = cache_dir / f"{h}_spatial.pt"
    
    # Skip if cache already exists
    if cache_file.exists():
        return 'skip'
    
    try:
        # ========== EXTRACT DCT BLOCKS ==========
        jpeg_struct = jio.read(str(img_path))
        coef_arrays = jpeg_struct.coef_arrays
        
        if coef_arrays is None or len(coef_arrays) == 0:
            return f'error: No DCT coefficients in {img_path.name}'
        
        # Extract blocks for each channel (Y, Cb, Cr)
        result = []
        for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
            if channel_name not in dct_channels:
                continue
            
            # Handle grayscale images (missing Cb/Cr)
            if channel_idx >= len(coef_arrays):
                if len(result) > 0:
                    # Use Y channel dimensions, fill with zeros
                    h_blocks, w_blocks, _, _ = result[0].shape
                    zero_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)
                    result.append(zero_blocks)
                else:
                    return f'error: Missing Y channel in {img_path.name}'
                continue
            
            # Extract DCT coefficients for this channel
            coeffs = coef_arrays[channel_idx]  # Shape: [H_blocks*8, W_blocks*8]
            
            # Reshape into 8×8 blocks: [H_blocks, W_blocks, 8, 8]
            h, w = coeffs.shape
            h_blocks = h // 8
            w_blocks = w // 8
            
            # Crop to multiple of 8
            coeffs_reshaped = coeffs[:h_blocks*8, :w_blocks*8]
            
            # Reshape: (H*8, W*8) → (H, 8, W, 8) → (H, W, 8, 8)
            coeffs_blocks = coeffs_reshaped.reshape(h_blocks, 8, w_blocks, 8).transpose(0, 2, 1, 3)
            
            # Convert to float32
            coeffs_blocks = coeffs_blocks.astype(np.float32)
            
            result.append(coeffs_blocks)
        
        # ========== CROP/PAD TO TARGET SIZE ==========
        processed_channels = []
        for blocks in result:
            blocks_h, blocks_w, _, _ = blocks.shape
            target_h, target_w = target_blocks
            
            # Create zero-padded output
            output = np.zeros((target_h, target_w, 8, 8), dtype=np.float32)
            
            # Center crop if larger, pad if smaller
            h_start = max(0, (blocks_h - target_h) // 2)
            w_start = max(0, (blocks_w - target_w) // 2)
            h_end = min(blocks_h, h_start + target_h)
            w_end = min(blocks_w, w_start + target_w)
            
            out_h = h_end - h_start
            out_w = w_end - w_start
            
            if out_h > 0 and out_w > 0:
                output[:out_h, :out_w, :, :] = blocks[h_start:h_end, w_start:w_end, :, :]
            
            processed_channels.append(output)
        
        # ========== SAVE TO CACHE ==========
        # Stack channels: [C, H, W, 8, 8]
        features_np = np.stack(processed_channels, axis=0).astype(np.float32)
        features = torch.from_numpy(features_np)
        
        # Save atomically (write to temp file, then rename)
        tmp_cache = cache_file.with_suffix('.tmp')
        torch.save({'features': features}, tmp_cache)
        tmp_cache.replace(cache_file)
        
        return 'success'
        
    except Exception as e:
        return f'error: {e}'


def main():
    """Precompute DCT features for entire dataset using multiprocessing."""
    
    logger.info("="*70)
    logger.info("DCT SPATIAL FEATURE PRECOMPUTATION")
    logger.info("="*70)
    logger.info("Excel:         %s", EXCEL_PATH)
    logger.info("Cache dir:     %s", CACHE_DIR)
    logger.info("Target blocks: %s (= %dx%d pixels)", TARGET_BLOCKS, TARGET_BLOCKS[0]*8, TARGET_BLOCKS[1]*8)
    logger.info("DCT channels:  %s", DCT_CHANNELS)
    logger.info("Workers:       %d", NUM_WORKERS)
    logger.info("="*70)
    
    # Validate Excel exists
    if not EXCEL_PATH.exists():
        logger.error("Excel file not found: %s", EXCEL_PATH)
        logger.error("Run combine_boss_rgb.py first to create combined dataset!")
        return
    
    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset to get image count
    logger.info("\nLoading dataset...")
    try:
        dataset = HybridSpatialDCTDataset(
            excel_path=EXCEL_PATH,
            img_root=IMG_ROOT,
            dct_channels=DCT_CHANNELS,
            target_blocks=TARGET_BLOCKS,
            cache_dir=CACHE_DIR,
            train=False  # No augmentation during caching
        )
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        return
    
    total_images = len(dataset)
    logger.info("Found %d images in dataset", total_images)
    
    if total_images == 0:
        logger.error("Dataset is empty!")
        return
    
    # Build configuration for workers
    dataset_config = {
        'excel_path': str(EXCEL_PATH),
        'img_root': str(IMG_ROOT),
        'dct_channels': DCT_CHANNELS,
        'target_blocks': list(TARGET_BLOCKS),
        'cache_dir': str(CACHE_DIR),
        'path_col': dataset.path_col,
    }
    
    # Create work items (one per image)
    worker_args = [(dataset_config, idx) for idx in range(total_images)]
    
    # Process images in parallel
    logger.info("\nProcessing %d images with %d workers...", total_images, NUM_WORKERS)
    
    start_time = time.time()
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    with Pool(NUM_WORKERS) as pool:
        results = pool.imap_unordered(process_image, worker_args, chunksize=10)
        
        for result in tqdm(results, total=total_images, desc="Caching DCT features"):
            if result == 'success':
                success_count += 1
            elif result == 'skip':
                skipped_count += 1
            else:
                # Error message
                failed_count += 1
                if failed_count <= 10:  # Only log first 10 errors
                    logger.warning(result)
    
    elapsed = time.time() - start_time
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PRECOMPUTATION COMPLETE")
    logger.info("="*70)
    logger.info("Time elapsed:      %.1f seconds (%.1f images/sec)", elapsed, total_images/elapsed)
    logger.info("Successfully cached: %d", success_count)
    logger.info("Skipped (existing):  %d", skipped_count)
    logger.info("Failed:              %d", failed_count)
    logger.info("Total processed:     %d", total_images)
    logger.info("="*70)
    logger.info("Cache directory:     %s", CACHE_DIR)
    logger.info("Cache size:          %d files", len(list(CACHE_DIR.glob('*_spatial.pt'))))
    logger.info("="*70)
    
    if failed_count > 0:
        logger.warning("\n%d images failed to cache. Check logs above for details.", failed_count)
        logger.warning("Training will still work, but may be slower for failed images.")
    
    logger.info("\n✓ Ready for training! Run train_hybrid.py")


if __name__ == "__main__":
    main()
