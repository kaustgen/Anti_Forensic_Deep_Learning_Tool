# Author: Kaleb Austgen
# Date: 10/20/25
# Purpose: Run this before sten_dct_train to precompute all features so epoch training is quick

from pathlib import Path
from sten_dct import StegoImageDataset
import logging

logging.basicConfig(level=logging.INFO)

EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'
IMG_ROOT = Path('.')

print("Loading dataset...")
dataset = StegoImageDataset(
    EXCEL_PATH, 
    IMG_ROOT, 
    dct_channels=['Y', 'Cb', 'Cr']
)

print(f"Pre-computing features for {len(dataset)} images...")
dataset.precompute_cache()
print("Pre-Compute finished")