# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose: Run this before sten_dct_cnn_train to precompute all spatial DCT features
#          Uses multiprocessing for speed (10x faster than sequential)

from pathlib import Path
from sten_dct_cnn import StegoDCTSpatialDataset
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)

EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'
IMG_ROOT = Path('.')
NUM_WORKERS = max(1, cpu_count() - 1)  # Use all CPUs minus 1


def process_image(args):
    """Worker function to cache a single image"""
    dataset_config, idx = args
    import torch
    import numpy as np
    import jpegio as jio
    import pandas as pd
    from hashlib import sha1
    from pathlib import Path
    
    # Reconstruct dataset info
    excel_path = dataset_config['excel_path']
    img_root = Path(dataset_config['img_root'])
    dct_channels = dataset_config['dct_channels']
    target_blocks = tuple(dataset_config['target_blocks'])
    cache_dir = Path(dataset_config['cache_dir'])
    path_col = dataset_config['path_col']
    
    # Load this row
    records = pd.read_excel(excel_path, header=0)
    row = records.iloc[idx]
    img_path = Path(row[path_col])
    if not img_path.is_absolute():
        img_path = (img_root / img_path).resolve()
    
    # Check cache
    h = sha1(str(img_path).encode('utf-8')).hexdigest()
    cache_file = cache_dir / f"{h}_spatial.pt"
    if cache_file.exists():
        return 'skip'
    
    try:
        # Extract DCT blocks
        jpeg_struct = jio.read(str(img_path))
        coef_arrays = jpeg_struct.coef_arrays
        
        result = []
        for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
            if channel_name not in dct_channels or channel_idx >= len(coef_arrays):
                continue
            
            coeffs = coef_arrays[channel_idx]
            h, w = coeffs.shape
            h_blocks, w_blocks = h // 8, w // 8
            coeffs_reshaped = coeffs[:h_blocks*8, :w_blocks*8]
            coeffs_blocks = coeffs_reshaped.reshape(h_blocks, 8, w_blocks, 8).transpose(0, 2, 1, 3)
            result.append(coeffs_blocks)
        
        # Crop/pad to target size
        processed_channels = []
        for blocks in result:
            blocks_h, blocks_w, _, _ = blocks.shape
            target_h, target_w = target_blocks
            output = np.zeros((target_h, target_w, 8, 8), dtype=blocks.dtype)
            
            h_start = max(0, (blocks_h - target_h) // 2)
            w_start = max(0, (blocks_w - target_w) // 2)
            h_end = min(blocks_h, h_start + target_h)
            w_end = min(blocks_w, w_start + target_w)
            
            out_h = h_end - h_start
            out_w = w_end - w_start
            output[:out_h, :out_w, :, :] = blocks[h_start:h_end, w_start:w_end, :, :]
            processed_channels.append(output)
        
        # Save cache
        features_np = np.stack(processed_channels, axis=0).astype(np.float32)
        features = torch.from_numpy(features_np)
        tmp_cache = cache_file.with_suffix('.tmp')
        torch.save({'features': features}, tmp_cache)
        tmp_cache.replace(cache_file)
        
        return 'success'
    except Exception as e:
        return f'error: {e}'


print("Loading dataset...")
dataset = StegoDCTSpatialDataset(
    EXCEL_PATH,
    IMG_ROOT,
    dct_channels=['Y', 'Cb', 'Cr'],
    target_blocks=(32, 32)  # 256x256 pixels
)

print(f"Pre-computing spatial DCT features for {len(dataset)} images using {NUM_WORKERS} workers...")

# Build config for workers
dataset_config = {
    'excel_path': str(EXCEL_PATH),
    'img_root': str(IMG_ROOT),
    'dct_channels': dataset.dct_channels,
    'target_blocks': list(dataset.target_blocks),
    'cache_dir': str(dataset.cache_dir),
    'path_col': dataset.path_col,
}

worker_args = [(dataset_config, idx) for idx in range(len(dataset))]

start_time = time.time()
success = 0
skipped = 0
failed = 0

with Pool(NUM_WORKERS) as pool:
    results = pool.imap_unordered(process_image, worker_args, chunksize=10)
    for result in tqdm(results, total=len(dataset), desc="Caching"):
        if result == 'success':
            success += 1
        elif result == 'skip':
            skipped += 1
        else:
            failed += 1

elapsed = time.time() - start_time

print(f"\nPre-Compute finished in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} images/sec)")
print(f"  Success: {success}")
print(f"  Skipped: {skipped} (already cached)")
print(f"  Failed:  {failed}")
print(f"\nCache directory: {dataset.cache_dir}")
print("Ready for training!")
