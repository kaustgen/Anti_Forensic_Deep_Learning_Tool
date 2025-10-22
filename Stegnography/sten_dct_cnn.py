# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose: True CNN for steganography detection using raw DCT coefficients as spatial input

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import logging
from hashlib import sha1

# Try to import jpegio for native DCT extraction
try:
    import jpegio as jio
    HAVE_JPEGLIB = True
except Exception:
    jio = None
    HAVE_JPEGLIB = False

logger = logging.getLogger(__name__)


class StegoDCTSpatialDataset(Dataset):
    """
    Dataset that returns raw DCT coefficients as 3D tensors (channels × height × width).
    
    Instead of summarizing DCT blocks into statistical features, this dataset
    preserves the spatial structure of 8×8 DCT blocks, allowing CNNs to learn
    local patterns across neighboring blocks.
    
    Output shape: [num_channels, num_blocks_h, num_blocks_w, 8, 8]
    where each 8×8 subarray is a DCT block from the JPEG.
    """
    
    def __init__(self, excel_path, img_root: Path, dct_channels=None, target_blocks=(32, 32), cache_dir=None):
        """
        Args:
            excel_path: Path to Excel with [image_path, is_stego] columns
            img_root: Base directory for image paths
            dct_channels: Which YCbCr channels to use (default: ['Y', 'Cb', 'Cr'])
            target_blocks: Target grid size (H, W) in 8×8 blocks. Images will be
                          cropped/padded to this size. E.g., (32, 32) = 256×256 pixels
            cache_dir: Directory for caching extracted DCT arrays
        """
        # Load Excel
        self.records = pd.read_excel(excel_path, header=0)
        logger.info("Loaded %d records from %s", len(self.records), excel_path)
        
        if len(self.records.columns) < 2:
            raise ValueError(f"Excel must have >=2 columns (path, label). Found: {list(self.records.columns)}")
        
        self.path_col = self.records.columns[0]
        self.label_col = self.records.columns[1]
        
        # Convert labels to boolean
        if self.records[self.label_col].dtype == 'object':
            self.records[self.label_col] = self.records[self.label_col].map({
                'True': True, 'TRUE': True, True: True,
                'False': False, 'FALSE': False, False: False
            })
        
        self.img_root = Path(img_root) if img_root else Path('.')
        self.dct_channels = dct_channels or ['Y', 'Cb', 'Cr']
        self.target_blocks = target_blocks  # (H, W) in blocks
        
        # Cache directory for DCT arrays
        default_cache = Path(__file__).parent / '.dct_spatial_cache'
        self.cache_dir = (Path(cache_dir) if cache_dir else default_cache).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate and normalize paths
        kept_rows = []
        dropped = 0
        for i, row in self.records.iterrows():
            raw_path = row[self.path_col]
            if pd.isna(raw_path):
                logger.warning("Dropping row %d: empty path", i)
                dropped += 1
                continue
            
            p = Path(raw_path)
            if not p.is_absolute():
                p = (self.img_root / p).resolve()
            
            if not p.exists():
                logger.warning("Dropping row %d: file not found: %s", i, p)
                dropped += 1
                continue
            
            kept_rows.append({self.path_col: str(p), self.label_col: row[self.label_col]})
        
        if dropped:
            logger.info("Dropped %d invalid/missing records", dropped)
        
        if len(kept_rows) == 0:
            raise RuntimeError(f"No valid image paths found in {excel_path}")
        
        self.records = pd.DataFrame(kept_rows).reset_index(drop=True)
        
        logger.info("Dataset: %d images, target_blocks=%s, channels=%s", 
                   len(self.records), target_blocks, self.dct_channels)
    
    def _cache_path_for(self, img_path: Path):
        """Generate cache file path based on image path hash"""
        h = sha1(str(img_path).encode('utf-8')).hexdigest()
        return self.cache_dir / f"{h}_spatial.pt"
    
    def _extract_dct_blocks(self, img_path: Path):
        """
        Extract raw DCT coefficient blocks from JPEG.
        
        Returns:
            List of 3D arrays, one per channel: [num_blocks_h, num_blocks_w, 8, 8]
            where each [8, 8] subarray is a DCT block.
        """
        if not HAVE_JPEGLIB:
            raise RuntimeError("jpegio not available. Install it with: pip install jpegio")
        
        try:
            # Read JPEG structure
            jpeg_struct = jio.read(str(img_path))
            coef_arrays = jpeg_struct.coef_arrays
            
            if coef_arrays is None or len(coef_arrays) == 0:
                raise ValueError(f"No DCT coefficients found in {img_path}")
            
            # coef_arrays is a list of 2D arrays (one per channel in YCbCr order)
            # Each array has shape [H, W] where H and W are already in 8×8 blocks
            
            result = []
            for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
                if channel_name not in self.dct_channels:
                    continue
                
                if channel_idx >= len(coef_arrays):
                    logger.warning("Channel %s not found in %s", channel_name, img_path)
                    continue
                
                coeffs = coef_arrays[channel_idx]  # Shape: [H_blocks*8, W_blocks*8]
                
                # Reshape into blocks: [H_blocks, W_blocks, 8, 8]
                h, w = coeffs.shape
                h_blocks = h // 8
                w_blocks = w // 8
                
                # Reshape: (H_blocks*8, W_blocks*8) → (H_blocks, 8, W_blocks, 8) → (H_blocks, W_blocks, 8, 8)
                coeffs_reshaped = coeffs[:h_blocks*8, :w_blocks*8]
                coeffs_blocks = coeffs_reshaped.reshape(h_blocks, 8, w_blocks, 8).transpose(0, 2, 1, 3)
                
                result.append(coeffs_blocks)
            
            return result
        
        except Exception as e:
            logger.error("Failed to extract DCT from %s: %s", img_path, e)
            raise
    
    def _crop_or_pad_blocks(self, blocks_3d):
        """
        Crop or pad block array to target_blocks size.
        
        Args:
            blocks_3d: Array of shape [H_blocks, W_blocks, 8, 8]
        
        Returns:
            Array of shape [target_H, target_W, 8, 8]
        """
        h, w, _, _ = blocks_3d.shape
        target_h, target_w = self.target_blocks
        
        # Create output array (zero-padded)
        output = np.zeros((target_h, target_w, 8, 8), dtype=blocks_3d.dtype)
        
        # Copy available blocks (center crop if larger, pad if smaller)
        h_start = max(0, (h - target_h) // 2)
        w_start = max(0, (w - target_w) // 2)
        
        h_end = min(h, h_start + target_h)
        w_end = min(w, w_start + target_w)
        
        out_h = h_end - h_start
        out_w = w_end - w_start
        
        output[:out_h, :out_w, :, :] = blocks_3d[h_start:h_end, w_start:w_end, :, :]
        
        return output
    
    def precompute_cache(self, max_workers=4, force=False):
        """
        Precompute and cache DCT features for all images in the dataset.
        
        Args:
            max_workers: Number of parallel workers (not implemented yet - sequential for safety)
            force: If True, recompute even if cache exists
        
        Returns:
            Dictionary with success/failure counts
        """
        from tqdm import tqdm
        
        logger.info("Precomputing DCT spatial cache for %d images...", len(self))
        
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for idx in tqdm(range(len(self)), desc="Caching DCT blocks"):
            row = self.records.iloc[idx]
            img_path = Path(row[self.path_col])
            if not img_path.is_absolute():
                img_path = (self.img_root / img_path).resolve()
            
            cache_file = self._cache_path_for(img_path)
            
            # Skip if cache exists and not forcing
            if cache_file.exists() and not force:
                skip_count += 1
                continue
            
            try:
                # Extract DCT blocks
                channel_blocks = self._extract_dct_blocks(img_path)
                
                # Crop/pad each channel to target size
                processed_channels = []
                for blocks in channel_blocks:
                    blocks_normalized = self._crop_or_pad_blocks(blocks)
                    processed_channels.append(blocks_normalized)
                
                # Stack channels: [C, H, W, 8, 8]
                features_np = np.stack(processed_channels, axis=0).astype(np.float32)
                features = torch.from_numpy(features_np)
                
                # Save to cache atomically
                tmp_cache = cache_file.with_suffix('.tmp')
                torch.save({'features': features}, tmp_cache)
                tmp_cache.replace(cache_file)
                
                success_count += 1
                
            except Exception as e:
                logger.error("Failed to precompute cache for %s: %s", img_path, e)
                fail_count += 1
                continue
        
        result = {
            'success': success_count,
            'skipped': skip_count,
            'failed': fail_count,
            'total': len(self)
        }
        
        logger.info("Cache precomputation complete: %s", result)
        return result
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: Tensor of shape [C, H, W, 8, 8] where:
                      C = number of channels (e.g., 3 for Y+Cb+Cr)
                      H, W = target_blocks dimensions
                      8, 8 = DCT block size
            label: 0 (clean) or 1 (stego)
        """
        row = self.records.iloc[idx]
        label = int(bool(row[self.label_col]))  # 0 or 1
        
        img_path = Path(row[self.path_col])
        if not img_path.is_absolute():
            img_path = (self.img_root / img_path).resolve()
        
        cache_file = self._cache_path_for(img_path)
        
        # Try loading from cache
        if cache_file.exists():
            try:
                cached = torch.load(cache_file)
                return cached['features'], label
            except Exception as e:
                logger.warning("Cache load failed for %s: %s, recomputing", img_path, e)
        
        # Extract DCT blocks
        try:
            channel_blocks = self._extract_dct_blocks(img_path)
        except Exception as e:
            logger.error("DCT extraction failed for %s: %s", img_path, e)
            # Return empty tensor (will be filtered by training code)
            empty = torch.zeros((len(self.dct_channels), *self.target_blocks, 8, 8), dtype=torch.float32)
            return empty, label
        
        # Crop/pad each channel to target size
        processed_channels = []
        for blocks in channel_blocks:
            blocks_normalized = self._crop_or_pad_blocks(blocks)
            processed_channels.append(blocks_normalized)
        
        # Stack channels: [C, H, W, 8, 8]
        features_np = np.stack(processed_channels, axis=0).astype(np.float32)
        features = torch.from_numpy(features_np)
        
        # Save to cache
        try:
            torch.save({'features': features}, cache_file)
        except Exception as e:
            logger.warning("Cache save failed for %s: %s", img_path, e)
        
        return features, label


class StegDetectionCNNSpatial(nn.Module):
    """
    True CNN architecture for steganography detection using spatial DCT blocks.
    
    Architecture:
        1. Reshape input from [B, C, H, W, 8, 8] to [B, C*64, H, W]
           (treat each 8×8 DCT block as 64 feature maps)
        2. Convolutional layers to learn spatial patterns across blocks
        3. Global pooling to aggregate spatial information
        4. Fully connected layers for classification
    """
    
    def __init__(self, num_channels=3, target_blocks=(32, 32), dropout=0.5):
        """
        Args:
            num_channels: Number of input channels (3 for Y+Cb+Cr)
            target_blocks: Input spatial dimensions in blocks (H, W)
            dropout: Dropout probability
        """
        super(StegDetectionCNNSpatial, self).__init__()
        
        self.num_channels = num_channels
        self.target_blocks = target_blocks
        
        # Input: [B, C, H, W, 8, 8] → reshape to [B, C*64, H, W]
        # where C*64 are the feature maps (each DCT coefficient becomes a channel)
        in_channels = num_channels * 64
        
        # Convolutional feature extraction
        # Use progressively larger receptive fields to capture block interactions
        self.conv_blocks = nn.Sequential(
            # Block 1: Local 3×3 patterns
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample: 32×32 → 16×16
            nn.Dropout2d(dropout * 0.3),
            
            # Block 2: Mid-level patterns
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample: 16×16 → 8×8
            nn.Dropout2d(dropout * 0.5),
            
            # Block 3: High-level patterns
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample: 8×8 → 4×4
            nn.Dropout2d(dropout * 0.7),
        )
        
        # Global average pooling to get fixed-size features regardless of input size
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary: clean vs stego
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W, 8, 8]
        
        Returns:
            Logits of shape [B, 2]
        """
        batch_size, num_channels, h, w, _, _ = x.shape
        
        # Reshape: [B, C, H, W, 8, 8] → [B, C*64, H, W]
        # This treats each of the 64 DCT coefficients in an 8×8 block as separate feature maps
        x = x.view(batch_size, num_channels * 64, h, w)
        
        # Apply convolutional layers
        x = self.conv_blocks(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        x = self.classifier(x)
        
        return x