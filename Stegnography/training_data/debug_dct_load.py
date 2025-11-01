#!/usr/bin/env python3
"""
Debug script to identify which image is causing the segfault.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sten_dct_cnn import StegoDCTSpatialDataset
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    excel_path = Path("../../dataGen/stego_training.xlsx")
    base_dir = excel_path.parent
    
    print(f"Loading dataset from {excel_path.resolve()}")
    print(f"Base directory: {base_dir.resolve()}")
    
    dataset = StegoDCTSpatialDataset(
        excel_path,
        base_dir,
        dct_channels=['Y', 'Cb', 'Cr'],
        target_blocks=(32, 32),
        train=False
    )
    
    print(f"\nDataset loaded: {len(dataset)} images")
    print("Testing first 10 samples...")
    
    for idx in range(min(10, len(dataset))):
        try:
            print(f"\n[{idx}] Loading sample {idx}...")
            img_path = dataset.records.iloc[idx][dataset.path_col]
            print(f"  Path: {img_path}")
            
            features, label = dataset[idx]
            print(f"  SUCCESS! Shape: {features.shape}, Label: {label}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nFailed on image {idx}: {dataset.records.iloc[idx]}")
            break

if __name__ == '__main__':
    main()
