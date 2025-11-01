# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose: Hybrid Spatial+Frequency CNN for steganography detection
#
# This module implements a sophisticated steganalysis architecture that combines:
#   1. Spatial Domain Processing: FilterBank HPF filters applied to RGB pixels
#   2. Frequency Domain Processing: DCT coefficient analysis from JPEG compression
#   3. Hybrid Fusion: Concatenates both feature streams for classification
#
# Key components:
#   - FilterBank: 15 hand-crafted HPF filters from steganalysis research
#   - StegoDCTSpatialDataset: Loads images + extracts DCT blocks from JPEGs
#   - SpatialFrequencyHybridCNN: Main model architecture
#   - HybridSpatialDCTDataset: Extended dataset for hybrid model
#
# Architecture inspired by:
#   - SRNet (2017): "Deep Residual Network for Steganalysis of Digital Images"
#   - XuNet (2016): "Structural Design of Convolutional Neural Networks for Steganalysis"
#   - Thesis: "Enhancing CNN performance using Spatial Rich Model and 2D Gabor filters"
#     https://bearworks.missouristate.edu/cgi/viewcontent.cgi?article=4762&context=theses

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import logging
from hashlib import sha1

# ============================================================
# JPEG DCT EXTRACTION LIBRARY
# ============================================================

# Try to import jpegio for native DCT extraction
# jpegio provides direct access to JPEG DCT coefficients without decoding
# This is critical for frequency domain steganalysis
try:
    import jpegio as jio
    HAVE_JPEGLIB = True
except Exception:
    jio = None
    HAVE_JPEGLIB = False

logger = logging.getLogger(__name__)

# ============================================================
# FILTERBANK: 15 HIGH-PASS FILTERS FOR SPATIAL STEGANALYSIS
# ============================================================

# These filters come from this Master's thesis: https://bearworks.missouristate.edu/cgi/viewcontent.cgi?article=4762&context=theses
# Implementation assisted by Claude Sonnet 4.5

class FilterBank(nn.Module):
    """
    15-filter bank for spatial steganalysis from research thesis.
    
    Filters are hand-crafted high-pass filters (HPFs) designed to highlight
    steganographic artifacts in the spatial domain. These filters detect:
        - Edge inconsistencies (directional edge filters)
        - Noise residuals (KV, KP filters)
        - Texture patterns (Gabor filters)
    
    Filter composition:
        - 1 KV (Square 5×5): Non-directional edge detection
        - 4 Edge 5×5: Directional edges (left, right, top, bottom)
        - 1 KP (Point filter): High-pass point filter
        - 1 KB (Edge detection): 3×3 edge filter
        - 8 2D Gabor: Oriented texture filters (4 angles × 2 phases)
    
    Reference: "Enhancing the performance of convolutional neural network 
               image-based steganalysis in spatial domain using Spatial Rich 
               Model and 2D Gabor filters"
    https://bearworks.missouristate.edu/cgi/viewcontent.cgi?article=4762&context=theses
    
    Usage:
        filterbank = FilterBank()
        filtered = filterbank(rgb_img)  # [B, 3, H, W] → [B, 45, H, W]
    """
    def __init__(self):
        super().__init__()

        filters = []

        # ========== FILTER 1: KV (Square 5x5) ==========
        # Non-directional edge detection filter
        # Detects edges regardless of orientation
        kv = np.array([
            [-1, 2, -2,  2, -1],
            [ 2,-6,  8, -6,  2],
            [-2, 8,-12,  8, -2],
            [ 2,-6,  8, -6,  2],
            [-1, 2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

        # Normalize: divide by absolute max
        kv = kv / np.abs(kv).max()
        filters.append(kv)

        # FILTERS 2-5: Edge 5x5 (4 Directions)
        edge_left = np.array([
            [-1, 2, -2,  0,  0],
            [ 2,-6,  8,  0,  0],
            [-2, 8,-12,  0,  0],
            [ 2,-6,  8,  0,  0],
            [-1, 2, -2,  0,  0]
        ], dtype=np.float32) / 12.0
        edge_left = edge_left / np.abs(edge_left).max()
        filters.append(edge_left)
        
        # Right (mirror of left)
        edge_right = np.array([
            [ 0,  0, -2,  2, -1],
            [ 0,  0,  8, -6,  2],
            [ 0,  0,-12,  8, -2],
            [ 0,  0,  8, -6,  2],
            [ 0,  0, -2,  2, -1]
        ], dtype=np.float32) / 12.0
        edge_right = edge_right / np.abs(edge_right).max()
        filters.append(edge_right)
        
        # Top
        edge_top = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32) / 12.0
        edge_top = edge_top / np.abs(edge_top).max()
        filters.append(edge_top)
        
        # Bottom (mirror of top)
        edge_bottom = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0
        edge_bottom = edge_bottom / np.abs(edge_bottom).max()
        filters.append(edge_bottom)

        # FILTER 6: KP (Point FIlter)
        # High-pass point filter (identity minus low-pass)
        kp = np.array([
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 24, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]
        ], dtype=np.float32) / 25.0
        kp = kp / np.abs(kp).max()
        filters.append(kp)

        # FILTER 7: KB (Edge Detection)
        # 3x3 edge filter padded to 5x5
        kb_3x3 = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32) / 4.0
        
        # Pad to 5×5
        kb = np.zeros((5, 5), dtype=np.float32)
        kb[1:4, 1:4] = kb_3x3
        kb = kb / np.abs(kb).max()
        filters.append(kb)

        # FILTERS 8-15: 2D Gabor (8 Filters)
        # Parameters from thesis:
        # σ = 1, θ = (0, π/2, π/4, 3π/4), ϕ = (π/4, π/8)
        # This creates 4 orientations × 2 phases = 8 filters

        orientations = [0, np.pi/2, np.pi/4, 3*np.pi/4] # 4 angles
        phases = [np.pi/4, np.pi/8] # 2 phases

        for phase in phases:
            for theta in orientations:
                gabor = self._create_gabor_filter(
                    ksize=5,
                    sigma=1.0,      # Fixed σ = 1
                    theta=theta,    # 4 orientations
                    lambd=2.0,      # Wavelength (not specified, using reasonable default)
                    gamma=0.5,      # From thesis equation
                    psi=phase       # 2 phases
                )
                # Make zero-mean (high-pass)
                gabor = gabor - gabor.mean()
                # Normalize
                gabor = gabor / np.abs(gabor).max()
                filters.append(gabor)
        
        # Convert all filters to tensors: [15, 1, 5, 5]
        filters_tensor = torch.stack([
            torch.from_numpy(f).unsqueeze(0).unsqueeze(0) 
            for f in filters
        ]).squeeze(1)  # [15, 1, 5, 5]

                # Register as non-trainable buffer
        self.register_buffer('weight', filters_tensor)
        
        print(f"Loaded {len(filters)} filters from thesis:")
        print(f"  - 1 Kv (Square 5×5)")
        print(f"  - 4 Edge 5×5 (left, right, top, bottom)")
        print(f"  - 1 Kp (point filter)")
        print(f"  - 1 Kb (edge detection)")
        print(f"  - 8 2D Gabor (4 orientations × 2 phases)")

    def _create_gabor_filter(self, ksize=5, sigma=1.0, theta=0, lambd=2.0, gamma=0.5, psi=0):
        """
        Create 2D Gabor filter using thesis equation (14):
        g(x,y) = exp(-(x'^2 + γ^2*y'^2)/(2*σ^2)) * cos(2π*x'/λ + ϕ)
        
        Where:
            x' = x*cos(θ) + y*sin(θ)
            y' = -x*sin(θ) + y*cos(θ)
        """

                # Create coordinate grid
        half = ksize // 2
        x = np.linspace(-half, half, ksize)
        y = np.linspace(-half, half, ksize)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        x_prime = X * np.cos(theta) + Y * np.sin(theta)
        y_prime = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gaussian envelope

        # CHECK IF exp OR e
        gaussian = np.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2 * sigma**2))
        
        # Cosine wave
        cosine = np.cos(2 * np.pi * x_prime / lambd + psi)
        
        # Gabor = Gaussian × Cosine
        gabor = gaussian * cosine
        
        return gabor.astype(np.float32)
    
    def forward(self, x):
        """
        Apply all 15 filters to input
        
        Args:
            x: [B, C, H, W] - e.g., [16, 192, 32, 32] for DCT spatial input
        
        Returns:
            [B, C*15, H, W] - 15 filter responses per channel
        """
        B, C, H, W = x.shape

        # APply filters to each channel separately
        outputs = []
        for c in range(C):
            # Extract single channel: [B, 1, H, W]
            x_c = x[:, c:c+1, :, :]

            # Apply all 15 filters: [B, 15, H, W]
            filtered = F.conv2d(x_c, self.weight, padding=2)

            outputs.append(filtered)
        
        # Concatenate: [B, C*15, H, W]
        return torch.cat(outputs, dim=1)


class StegoDCTSpatialDataset(Dataset):
    """
    Dataset that returns raw DCT coefficients as 3D tensors (channels × height × width).
    
    Instead of summarizing DCT blocks into statistical features, this dataset
    preserves the spatial structure of 8×8 DCT blocks, allowing CNNs to learn
    local patterns across neighboring blocks.
    
    Output shape: [num_channels, num_blocks_h, num_blocks_w, 8, 8]
    where each 8×8 subarray is a DCT block from the JPEG.
    """
    
    def __init__(self, excel_path, img_root: Path, dct_channels=None, target_blocks=(32, 32), cache_dir=None, train=False):
        """
        Args:
            excel_path: Path to Excel with [image_path, is_stego] columns
            img_root: Base directory for image paths
            dct_channels: Which YCbCr channels to use (default: ['Y', 'Cb', 'Cr'])
            target_blocks: Target grid size (H, W) in 8×8 blocks. Images will be
                          cropped/padded to this size. E.g., (32, 32) = 256×256 pixels
            cache_dir: Directory for caching extracted DCT arrays
            train: If True, apply data augmentation (flips)
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
        self.train = train  # Whether to apply data augmentation
        
        # Cache directory for DCT arrays
        default_cache = Path(__file__).parent / '.dct_BOSS_RGB_combined_cache'
        #default_cache = Path(__file__).parent / '.dct_spatial_cache'
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
            
            # Keep all columns, but update the path to absolute
            row_dict = row.to_dict()
            row_dict[self.path_col] = str(p)
            kept_rows.append(row_dict)
        
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
            Always returns a list matching self.dct_channels length.
        """
        if not HAVE_JPEGLIB:
            raise RuntimeError("jpegio not available. Install it with: pip install jpegio")
        
        try:
            # Ensure file path is string and file exists
            img_path_str = str(img_path)
            if not Path(img_path_str).exists():
                raise FileNotFoundError(f"Image not found: {img_path_str}")
            
            # Read JPEG structure
            jpeg_struct = jio.read(img_path_str)
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
                    # Grayscale image - Cb/Cr channels don't exist
                    # Create zero-filled blocks matching the Y channel size
                    if len(result) > 0:
                        # Use same dimensions as Y channel
                        h_blocks, w_blocks, _, _ = result[0].shape
                        zero_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)
                        result.append(zero_blocks)
                    else:
                        # Shouldn't happen (Y channel should always exist)
                        raise ValueError(f"Missing Y channel in {img_path}")
                    continue
                
                coeffs = coef_arrays[channel_idx]  # Shape: [H_blocks*8, W_blocks*8]
                
                # Reshape into blocks: [H_blocks, W_blocks, 8, 8]
                h, w = coeffs.shape
                h_blocks = h // 8
                w_blocks = w // 8
                
                # Reshape: (H_blocks*8, W_blocks*8) → (H_blocks, 8, W_blocks, 8) → (H_blocks, W_blocks, 8, 8)
                coeffs_reshaped = coeffs[:h_blocks*8, :w_blocks*8]
                coeffs_blocks = coeffs_reshaped.reshape(h_blocks, 8, w_blocks, 8).transpose(0, 2, 1, 3)
                
                # Convert to float32 immediately to avoid dtype issues later
                coeffs_blocks = coeffs_blocks.astype(np.float32)
                
                result.append(coeffs_blocks)
            
            # Ensure we always return the expected number of channels
            if len(result) != len(self.dct_channels):
                raise ValueError(f"Expected {len(self.dct_channels)} channels, got {len(result)} from {img_path}")
            
            return result
        
        except Exception as e:
            logger.error("Failed to extract DCT from %s: %s", img_path, e)
            # Return empty blocks instead of raising (allows training to continue)
            # Use target_blocks dimensions for consistent shape
            empty_blocks = [np.zeros((*self.target_blocks, 8, 8), dtype=np.float32) 
                          for _ in range(len(self.dct_channels))]
            return empty_blocks
    
    def _crop_or_pad_blocks(self, blocks_3d):
        """
        Crop or pad block array to target_blocks size.
        
        Args:
            blocks_3d: Array of shape [H_blocks, W_blocks, 8, 8]
        
        Returns:
            Array of shape [target_H, target_W, 8, 8]
        """
        # Validate input BEFORE trying to access shape
        if blocks_3d is None:
            raise ValueError("blocks_3d is None")
        if not isinstance(blocks_3d, np.ndarray):
            raise ValueError(f"blocks_3d must be numpy array, got {type(blocks_3d)}")
        
        # Get shape safely
        try:
            shape = blocks_3d.shape
            if len(shape) != 4:
                raise ValueError(f"blocks_3d must be 4D, got shape {shape}")
            h, w, bh, bw = shape
        except Exception as e:
            raise ValueError(f"Failed to unpack blocks_3d shape: {e}, shape={blocks_3d.shape if hasattr(blocks_3d, 'shape') else 'no shape'}")
        
        # Validate block dimensions
        if bh != 8 or bw != 8:
            raise ValueError(f"Expected 8x8 blocks, got {bh}x{bw}")
        
        target_h, target_w = self.target_blocks
        
        # Create output array (zero-padded) with explicit dtype
        output = np.zeros((target_h, target_w, 8, 8), dtype=np.float32)
        
        # Handle edge case: no blocks available
        if h == 0 or w == 0:
            return output
        
        # Copy available blocks (center crop if larger, pad if smaller)
        h_start = max(0, (h - target_h) // 2)
        w_start = max(0, (w - target_w) // 2)
        
        h_end = min(h, h_start + target_h)
        w_end = min(w, w_start + target_w)
        
        out_h = h_end - h_start
        out_w = w_end - w_start
        
        # Validate indices before slicing
        if out_h > 0 and out_w > 0:
            try:
                # Convert to float32 first to avoid dtype issues during slicing
                blocks_float = blocks_3d.astype(np.float32, copy=False)
                output[:out_h, :out_w, :, :] = blocks_float[h_start:h_end, w_start:w_end, :, :]
            except Exception as e:
                raise ValueError(f"Failed to copy blocks: {e}, shapes: output={output.shape}, blocks_3d={blocks_3d.shape}, slice=({h_start}:{h_end}, {w_start}:{w_end})")
        
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
                
                # Handle grayscale images (only Y channel available)
                # Pad with zeros for missing Cb, Cr channels
                if len(processed_channels) == 1 and len(self.dct_channels) > 1:
                    # Create zero-filled arrays for missing channels
                    zero_channel = np.zeros_like(processed_channels[0])
                    for _ in range(len(self.dct_channels) - 1):
                        processed_channels.append(zero_channel)
                
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
                features = cached['features']
                
                # Handle old cached grayscale images (before zero-padding fix)
                if features.shape[0] == 1 and len(self.dct_channels) > 1:
                    # Old cache had shape [1, H, W, 8, 8], need to pad to [3, H, W, 8, 8]
                    zero_channel = torch.zeros_like(features[0:1])
                    pad_channels = [features]
                    for _ in range(len(self.dct_channels) - 1):
                        pad_channels.append(zero_channel)
                    features = torch.cat(pad_channels, dim=0)
                
                return features, label
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
        
        # Validate we got the expected number of channels
        if len(channel_blocks) != len(self.dct_channels):
            logger.error("Channel count mismatch for %s: expected %d, got %d", 
                        img_path, len(self.dct_channels), len(channel_blocks))
            empty = torch.zeros((len(self.dct_channels), *self.target_blocks, 8, 8), dtype=torch.float32)
            return empty, label
        
        # Crop/pad each channel to target size
        processed_channels = []
        for idx, blocks in enumerate(channel_blocks):
            try:
                blocks_normalized = self._crop_or_pad_blocks(blocks)
                processed_channels.append(blocks_normalized)
            except Exception as e:
                logger.error("Failed to crop/pad blocks for channel %d in %s: %s", idx, img_path, e)
                logger.error("  blocks shape: %s, type: %s", 
                           blocks.shape if hasattr(blocks, 'shape') else 'no shape', 
                           type(blocks))
                # Use zero channel as fallback
                zero_channel = np.zeros((self.target_blocks[0], self.target_blocks[1], 8, 8), dtype=np.float32)
                processed_channels.append(zero_channel)
        
        # Validate all processed channels have correct shape
        for idx, ch in enumerate(processed_channels):
            expected_shape = (self.target_blocks[0], self.target_blocks[1], 8, 8)
            if ch.shape != expected_shape:
                logger.error("Channel %d has wrong shape: %s, expected %s", idx, ch.shape, expected_shape)
                processed_channels[idx] = np.zeros(expected_shape, dtype=np.float32)
        
        # Stack channels: [C, H, W, 8, 8]
        try:
            features_np = np.stack(processed_channels, axis=0)
            # Ensure float32 dtype
            if features_np.dtype != np.float32:
                features_np = features_np.astype(np.float32)
        except Exception as e:
            logger.error("Failed to stack channels for %s: %s", img_path, e)
            features_np = np.zeros((len(self.dct_channels), *self.target_blocks, 8, 8), dtype=np.float32)
        
        features = torch.from_numpy(features_np)
        
        # Apply data augmentation (only during training)
        if self.train:
            import random
            # Horizontal flip
            if random.random() < 0.5:
                features = torch.flip(features, dims=[2])  # Flip width dimension
            
            # Vertical flip
            if random.random() < 0.5:
                features = torch.flip(features, dims=[1])  # Flip height dimension
        
        # Save to cache
        try:
            torch.save({'features': features}, cache_file)
        except Exception as e:
            logger.warning("Cache save failed for %s: %s", img_path, e)
        
        return features, label


# Truncated linear unit class
# Acts as a neuron, and adds weights to certain items
# Theoretically can increase the likelihood of the model detecting 
# difficult steganalysis
class TLU(nn.Module):
    """Truncated Linear Unit"""
    def __init__(self, T=3.0):
        super().__init__()
        self.T = float(T)
    def forward(self, x):
        return x.clamp(min=-self.T, max=self.T)


class ChannelAttention(nn.Module):
    """
    Channel Attention mechanism - learns to weight channels.
    Helps model down-weight zero Cb/Cr channels in grayscale images.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.shape
        
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)

# Allow gradients to flow to ealier layers, granting a deeper network
class ResidualBlock(nn.Module):
    """Residual Block with two convs and skip connections"""
    def __init__(self, in_channels, out_channels, downsample=None, use_tlu=False, tlu_threshold=3.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Use TLu for early blocks
        if use_tlu:
            self.activation = TLU(T=tlu_threshold)
        else:
            self.activation = nn.GELU()
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity # Residual connection
        out = self.activation(out)
        return out

class SpatialAttention(nn.Module):
    """Learn which spatial locations are important"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Aggregate across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and learn spatial attention
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.conv(spatial)
        attention = self.sigmoid(spatial)

        return x * attention


class StegDetectionCNNDCT(nn.Module):
    """
    SRNet-inspired CNN for steganography detection in DCT frequency domain.
    
    Architecture (based on SRNet paper):
        1. Type 1 Layers (1-2): Noise residual extraction without pooling
           - Start with 64 filters, reduce to 16 before unpooled segment
        2. Type 2 Layers (3-7): Unpooled residual layers to preserve spatial detail
           - Keep full 32×32 resolution to detect subtle 1.92% DCT changes
        3. Type 3 Layers (8-11): Pooled residual layers for higher-level features
           - Progressive pooling: 32×32 → 16×16 → 8×8
        4. Type 4 Layer (12): Global average pooling + classifier
           - Statistical moments of feature maps → classification
    
    Key SRNet principles:
        - NO early pooling (keep spatial resolution for subtle patterns)
        - Residual connections throughout (better gradient flow)
        - Strategic channel management (64 → 16 → 64 → 128)
        - Global pooling instead of aggressive spatial reduction
    """
    
    def __init__(self, num_channels=3, target_blocks=(32, 32), dropout=0.2):
        """
        Args:
            num_channels: Number of input channels (3 for Y+Cb+Cr)
            target_blocks: Input spatial dimensions in blocks (H, W)
            dropout: Dropout probability
        """
        super(StegDetectionCNNDCT, self).__init__()
        
        self.num_channels = num_channels
        self.target_blocks = target_blocks
        
        # Input: [B, C, H, W, 8, 8] → reshape to [B, C*64, H, W]
        # Each DCT coefficient (0-63) becomes a separate channel
        # For YCbCr: 3 color channels × 64 DCT coeffs = 192 input channels
        in_channels = num_channels * 64  # 192 for YCbCr
        
        # ========== TYPE 1 LAYERS: Noise Residual Extraction (NO POOLING) ==========
        # Similar to SRNet Layers 1-2: Extract high-frequency patterns
        # 192 → 64 → 16 (reduce channels before unpooled segment)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # ========== TYPE 2 LAYERS: Unpooled Residual Layers (NO POOLING) ==========
        # SRNet Layers 3-7: Keep 32×32 spatial resolution to preserve subtle patterns
        # Critical for detecting 1.92% DCT coefficient changes
        
        self.layer3 = self._make_unpooled_layer(16, 16)
        self.layer4 = self._make_unpooled_layer(16, 16)
        self.layer5 = self._make_unpooled_layer(16, 16)
        self.layer6 = self._make_unpooled_layer(16, 16)
        self.layer7 = self._make_unpooled_layer(16, 16)
        
        # ========== TYPE 3 LAYERS: Pooled Residual Layers (WITH POOLING) ==========
        # SRNet Layers 8-11: Now safe to pool after learning spatial patterns
        
        self.layer8 = self._make_pooled_layer(16, 64, pool=True)  # 32×32 → 16×16
        self.layer9 = self._make_pooled_layer(64, 64, pool=False)
        self.layer10 = self._make_pooled_layer(64, 128, pool=True)  # 16×16 → 8×8
        self.layer11 = self._make_pooled_layer(128, 128, pool=False)
        
        # ========== TYPE 4 LAYER: Global Pooling + Classifier ==========
        # SRNet Layer 12: Statistical moments (global average) → classifier
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 8×8 → 1×1
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary: clean (0) vs stego (1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _make_unpooled_layer(self, in_channels, out_channels):
        """
        Type 2 Layer: Unpooled residual block (like SRNet Layers 3-7)
        Preserves spatial resolution for subtle pattern detection
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Residual connection (input = output since same dimensions)
        )
    
    def _make_pooled_layer(self, in_channels, out_channels, pool=False):
        """
        Type 3 Layer: Pooled residual block (like SRNet Layers 8-11)
        Includes optional 3×3 average pooling with stride 2
        """
        layers = []
        
        # First conv (may change channels)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second conv (maintains channels)
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Optional pooling (after residual connection)
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU activation"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through SRNet-inspired architecture.
        
        Args:
            x: Tensor of shape [B, C, H, W, 8, 8]
               B = batch size
               C = num_channels (3 for YCbCr)
               H, W = target_blocks (e.g., 32, 32)
               8, 8 = DCT block size
        
        Returns:
            Logits of shape [B, 2] for binary classification
        """
        batch_size, num_channels, h, w, _, _ = x.shape
        
        # Reshape: [B, C, H, W, 8, 8] → [B, C*64, H, W]
        # This flattens each 8×8 DCT block into 64 separate channels
        # For YCbCr: [B, 3, 32, 32, 8, 8] → [B, 192, 32, 32]
        x = x.view(batch_size, num_channels * 64, h, w)
        
        # Type 1: Noise residual extraction (NO pooling)
        # 192 → 64 → 16 channels, maintain 32×32 spatial resolution
        x = self.layer1(x)  # [B, 192, 32, 32] → [B, 64, 32, 32]
        x = self.layer2(x)  # [B, 64, 32, 32] → [B, 16, 32, 32]
        
        # Type 2: Unpooled residual layers (NO pooling)
        # Keep 32×32 resolution to preserve subtle 1.92% DCT changes
        # Residual connections help gradient flow through 5 layers
        identity = x
        x = self.layer3(x) + identity  # [B, 16, 32, 32] (with residual)
        
        identity = x
        x = self.layer4(x) + identity  # [B, 16, 32, 32] (with residual)
        
        identity = x
        x = self.layer5(x) + identity  # [B, 16, 32, 32] (with residual)
        
        identity = x
        x = self.layer6(x) + identity  # [B, 16, 32, 32] (with residual)
        
        identity = x
        x = self.layer7(x) + identity  # [B, 16, 32, 32] (with residual)
        
        # Type 3: Pooled residual layers (WITH pooling)
        # Now safe to reduce spatial dimensions after learning patterns
        x = self.layer8(x)  # [B, 16, 32, 32] → [B, 64, 16, 16] (pooled)
        
        identity = F.avg_pool2d(x, kernel_size=1)  # Match dimensions for residual
        x = self.layer9(x) + identity  # [B, 64, 16, 16] (with residual)
        
        x = self.layer10(x)  # [B, 64, 16, 16] → [B, 128, 8, 8] (pooled)
        
        identity = F.avg_pool2d(x, kernel_size=1)  # Match dimensions for residual
        x = self.layer11(x) + identity  # [B, 128, 8, 8] (with residual)
        
        # Type 4: Global pooling + classification
        x = self.global_pool(x)  # [B, 128, 8, 8] → [B, 128, 1, 1]
        x = self.classifier(x)  # [B, 128, 1, 1] → [B, 2]
        
        return x
    
    # def forward(self, x):
    #     """
    #     Forward pass through the simplified CNN.
        
    #     Args:
    #         x: Tensor of shape [B, C, H, W, 8, 8]
    #            B = batch size
    #            C = num_channels (3 for YCbCr)
    #            H, W = target_blocks (e.g., 32, 32)
    #            8, 8 = DCT block size
        
    #     Returns:
    #         Logits of shape [B, 2] for binary classification
    #     """
    #     batch_size, num_channels, h, w, _, _ = x.shape
        
    #     # Reshape: [B, C, H, W, 8, 8] → [B, C*64, H, W]
    #     # This flattens each 8×8 DCT block into 64 separate channels
    #     # For YCbCr: [B, 3, 32, 32, 8, 8] → [B, 192, 32, 32]
    #     x = x.view(batch_size, num_channels * 64, h, w)
        
    #     # Normalize DCT coefficients
    #     #x = self.input_bn(x)
        
    #     # Apply channel attention (learn which DCT frequencies are important)
    #     #x = self.channel_attention(x)
        
    #     # Convolutional feature extraction
    #     x = self.block1(x)  # [B, 192, 32, 32] → [B, 128, 16, 16]
    #     x = self.block2(x)  # [B, 128, 16, 16] → [B, 64, 8, 8]
    #     x = self.block3(x)  # [B, 64, 8, 8] → [B, 64, 1, 1]
        
    #     # Classification
    #     x = self.classifier(x)  # [B, 64, 1, 1] → [B, 2]
        
    #     return x


# ============================================================
# HYBRID SPATIAL + FREQUENCY CNN
# ============================================================

class SpatialFrequencyHybridCNN(nn.Module):
    """
    True SRNet-style hybrid: Spatial HPF (on pixels) + DCT frequency features.
    
    Architecture:
        Spatial: [B, 3, 256, 256] pixels → FilterBank → [B, 45, 256, 256] 
                 → CNNs → [B, 256] spatial features
        
        DCT: [B, 3, 32, 32, 8, 8] coeffs → SRNet layers → [B, 128] DCT features
        
        Fusion: [B, 384] → FC layers → [B, 2] logits
    """
    
    def __init__(self, num_channels=3, target_blocks=(32, 32), dropout=0.3):
        super().__init__()
        
        self.num_channels = num_channels
        self.target_blocks = target_blocks
        self.diagnostic_mode = False  # Enable with model.diagnostic_mode = True
        
        # ========== SPATIAL BRANCH (HPF on Pixels) ==========
        # Apply FilterBank HPF filters to spatial pixels
        self.hpf_filters = FilterBank()  # 15 filters (Kv, Edge, Kp, Kb, Gabor)

        # InstanceNOrm layer to stablilize output of the HPF
        self.hpf_norm = nn.InstanceNorm2d(45, affine=True)
        
        # Freeze HPF filters (hand-crafted, don't train)
        for param in self.hpf_filters.parameters():
            param.requires_grad = False
        
        # Process spatial HPF features with SRNet-style architecture (11 layers, same as DCT branch)
        # Input: [B, 3*15=45, 256, 256] (3 color channels × 15 filters)
        # Output: [B, 256] spatial feature vector
        
        # Type 1: Noise residual extraction (NO pooling) - Layers 1-2
        # 45 → 64 → 16 (reduce channels before unpooled segment)
        self.spatial_layer1 = nn.Sequential(
            nn.Conv2d(45, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            TLU(T=3.0)
            #nn.ReLU(inplace=True)
        )
        
        self.spatial_layer2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            TLU(T=3.0)
            #nn.ReLU(inplace=True)
        )
        
        # Type 2: Unpooled residual layers (NO pooling) - Layers 3-7
        # Keep 256×256 resolution to preserve spatial detail
        self.spatial_layer3 = self._make_spatial_unpooled_layer(16, 16)
        self.spatial_layer4 = self._make_spatial_unpooled_layer(16, 16)
        self.spatial_layer5 = self._make_spatial_unpooled_layer(16, 16)
        self.spatial_layer6 = self._make_spatial_unpooled_layer(16, 16)
        self.spatial_layer7 = self._make_spatial_unpooled_layer(16, 16)
        
        # Type 3: Pooled residual layers (WITH pooling) - Layers 8-11
        # Progressive pooling: 256×256 → 128×128 → 64×64 → 32×32
        self.spatial_layer8 = self._make_spatial_pooled_layer(16, 64, pool=True)   # 256×256 → 128×128
        self.spatial_layer9 = self._make_spatial_pooled_layer(64, 64, pool=False)
        self.spatial_layer10 = self._make_spatial_pooled_layer(64, 256, pool=True)  # 128×128 → 64×64
        self.spatial_layer11 = self._make_spatial_pooled_layer(256, 256, pool=False)
        
        # Type 4: Global pooling - Layer 12
        self.spatial_global_pool = nn.AdaptiveAvgPool2d(1)  # Any size → 1×1
        
        # ========== DCT BRANCH (Frequency Domain) ==========
        #DCT blocks are normalized
        #self.dct_norm = nn.InstanceNorm2d(num_channels * 64, affine=False)
        # Existing SRNet-inspired DCT processor
        self.dct_branch = StegDetectionCNNDCT(
            num_channels=num_channels,
            target_blocks=target_blocks,
            dropout=0.0  # No dropout in branch (will apply in fusion)
        )
        
        # DCT feature dimension (from layer11 before classifier)
        self.dct_feature_dim = 128
        
        # ========== FUSION LAYER ==========
        # Combine spatial (256) + DCT (128) features
        fusion_input_dim = 256 + self.dct_feature_dim
        
        self.fusion_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, 2)  # Binary: clean (0) vs stego (1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Apply stricter initialization to final classifier layer to prevent explosion
        final_layer = self.fusion_classifier[-1]  # Last Linear layer
        if isinstance(final_layer, nn.Linear):
            nn.init.xavier_uniform_(final_layer.weight, gain=0.5)  # Smaller gain = smaller weights
            if final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, 0)
    
    def _make_spatial_unpooled_layer(self, in_channels, out_channels):
        """
        Type 2 Layer for spatial branch: Unpooled residual block
        Same structure as DCT branch to match capacity
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def _make_spatial_pooled_layer(self, in_channels, out_channels, pool=False):
        """
        Type 3 Layer for spatial branch: Pooled residual block
        Same structure as DCT branch to match capacity
        """
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Optional pooling
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_dct_features(self, dct_blocks):
        """
        Extract features from DCT branch (without classification).
        
        Args:
            dct_blocks: [B, C, H, W, 8, 8] DCT coefficients
        
        Returns:
            [B, 128] DCT features
        """
        B, C, H, W, _, _ = dct_blocks.shape
        
        # Reshape DCT blocks to channels
        x = dct_blocks.view(B, C * 64, H, W)

        # Normalize DCT coefficients
        #x = self.dct_norm(x)
        
        # Identify DC channel indices (0, 64, 128 for YCbCr)
        dc_indices = torch.arange(0, C * 64, 64, device=x.device)
        
        # Create masks
        dc_mask = torch.zeros(C * 64, dtype=torch.bool, device=x.device)
        dc_mask[dc_indices] = True
        ac_mask = ~dc_mask
        
        # Extract DC and AC channels
        dc_channels = x[:, dc_mask, :, :]  # [B, C, H, W]
        ac_channels = x[:, ac_mask, :, :]  # [B, C*63, H, W]
        
        # **Z-score normalization for DC** (per-image normalization)
        # Compute mean and std across spatial dimensions (H, W)
        dc_mean = dc_channels.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        dc_std = dc_channels.std(dim=[2, 3], keepdim=True) + 1e-8  # Avoid division by zero
        dc_normalized = (dc_channels - dc_mean) / dc_std
        
        # **Z-score normalization for AC** (per-image normalization)
        ac_mean = ac_channels.mean(dim=[2, 3], keepdim=True)  # [B, C*63, 1, 1]
        ac_std = ac_channels.std(dim=[2, 3], keepdim=True) + 1e-8
        ac_normalized = (ac_channels - ac_mean) / ac_std
        
        # **Clamp to prevent extreme outliers** (±6 std deviations instead of ±3)
        # This preserves 99.7% of data while preventing explosions
        dc_normalized = dc_normalized.clamp(-6.0, 6.0)
        ac_normalized = ac_normalized.clamp(-6.0, 6.0)
        
        # Reconstruct full tensor
        x_normalized = torch.zeros_like(x)
        x_normalized[:, dc_mask, :, :] = dc_normalized
        x_normalized[:, ac_mask, :, :] = ac_normalized
        
        # Pass through DCT branch layers (without final classifier)
        x = self.dct_branch.layer1(x_normalized)
        x = self.dct_branch.layer2(x)
        
        # Unpooled residual layers
        identity = x
        x = self.dct_branch.layer3(x) + identity
        identity = x
        x = self.dct_branch.layer4(x) + identity
        identity = x
        x = self.dct_branch.layer5(x) + identity
        identity = x
        x = self.dct_branch.layer6(x) + identity
        identity = x
        x = self.dct_branch.layer7(x) + identity
        
        # Pooled residual layers
        x = self.dct_branch.layer8(x)
        identity = F.avg_pool2d(x, kernel_size=1)
        x = self.dct_branch.layer9(x) + identity
        
        x = self.dct_branch.layer10(x)
        identity = F.avg_pool2d(x, kernel_size=1)
        x = self.dct_branch.layer11(x) + identity
        
        # Global pooling
        x = self.dct_branch.global_pool(x)  # [B, 128, 1, 1]
        x = x.view(B, -1)  # [B, 128]
        
        return x
    
    def forward(self, spatial_img, dct_blocks):
        """
        Forward pass through hybrid architecture.
        
        Args:
            spatial_img: [B, 3, H_pixels, W_pixels] - Decompressed RGB pixels
                        E.g., [B, 3, 256, 256] for 32×32 blocks
            dct_blocks: [B, C, H_blocks, W_blocks, 8, 8] - DCT coefficients
                       E.g., [B, 3, 32, 32, 8, 8]
        
        Returns:
            [B, 2] logits for binary classification
        """
        B = spatial_img.shape[0]
        
        if self.diagnostic_mode:
            print("\n" + "="*70)
            print("LAYER-BY-LAYER DIAGNOSTIC TRACKING")
            print("="*70)
        
        # ========== SPATIAL BRANCH ==========
        # Apply HPF filters to pixels: [B, 3, 256, 256] → [B, 45, 256, 256]
        spatial_filtered = self.hpf_filters(spatial_img)
        if self.diagnostic_mode:
            print(f"After HPF filters:     [{spatial_filtered.min():.3f}, {spatial_filtered.max():.3f}]")

        # Normalize the filter outputs to have zero mean and unit variance
        spatial_filtered = self.hpf_norm(spatial_filtered)
        if self.diagnostic_mode:
            print(f"After HPF norm:        [{spatial_filtered.min():.3f}, {spatial_filtered.max():.3f}]")
        
        # Process through 11-layer SRNet-style architecture (matches DCT branch)
        # Type 1: Noise extraction (NO pooling)
        x_spatial = self.spatial_layer1(spatial_filtered)  # [B, 45, 256, 256] → [B, 64, 256, 256]
        if self.diagnostic_mode:
            print(f"After spatial_layer1:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        x_spatial = self.spatial_layer2(x_spatial)         # [B, 64, 256, 256] → [B, 16, 256, 256]
        if self.diagnostic_mode:
            print(f"After spatial_layer2:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        # Type 2: Unpooled residual layers (preserve spatial detail)
        identity = x_spatial
        x_spatial = self.spatial_layer3(x_spatial) + identity  # [B, 16, 256, 256] + residual
        if self.diagnostic_mode:
            print(f"After spatial_layer3:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        identity = x_spatial
        x_spatial = self.spatial_layer4(x_spatial) + identity
        if self.diagnostic_mode:
            print(f"After spatial_layer4:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        identity = x_spatial
        x_spatial = self.spatial_layer5(x_spatial) + identity
        if self.diagnostic_mode:
            print(f"After spatial_layer5:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        identity = x_spatial
        x_spatial = self.spatial_layer6(x_spatial) + identity
        if self.diagnostic_mode:
            print(f"After spatial_layer6:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        identity = x_spatial
        x_spatial = self.spatial_layer7(x_spatial) + identity
        if self.diagnostic_mode:
            print(f"After spatial_layer7:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        # Type 3: Pooled residual layers (hierarchical features)
        x_spatial = self.spatial_layer8(x_spatial)  # [B, 16, 256, 256] → [B, 64, 128, 128] (pooled)
        if self.diagnostic_mode:
            print(f"After spatial_layer8:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        identity = F.avg_pool2d(x_spatial, kernel_size=1)
        x_spatial = self.spatial_layer9(x_spatial) + identity  # [B, 64, 128, 128] + residual
        if self.diagnostic_mode:
            print(f"After spatial_layer9:  [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        x_spatial = self.spatial_layer10(x_spatial)  # [B, 64, 128, 128] → [B, 256, 64, 64] (pooled)
        if self.diagnostic_mode:
            print(f"After spatial_layer10: [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        identity = F.avg_pool2d(x_spatial, kernel_size=1)
        x_spatial = self.spatial_layer11(x_spatial) + identity  # [B, 256, 64, 64] + residual
        if self.diagnostic_mode:
            print(f"After spatial_layer11: [{x_spatial.min():.3f}, {x_spatial.max():.3f}]")
        
        # Type 4: Global pooling
        x_spatial = self.spatial_global_pool(x_spatial)  # [B, 256, 64, 64] → [B, 256, 1, 1]
        spatial_features = x_spatial.view(B, -1)  # [B, 256]
        if self.diagnostic_mode:
            print(f"Spatial features:      [{spatial_features.min():.3f}, {spatial_features.max():.3f}] shape={spatial_features.shape}")
        
        # ========== DCT BRANCH ==========
        # Extract DCT features: [B, 3, 32, 32, 8, 8] → [B, 128]
        dct_features = self.extract_dct_features(dct_blocks)
        if self.diagnostic_mode:
            print(f"DCT features:          [{dct_features.min():.3f}, {dct_features.max():.3f}] shape={dct_features.shape}")
        
        # ========== FUSION ==========
        # Concatenate: [B, 256] + [B, 128] → [B, 384]
        combined = torch.cat([spatial_features, dct_features], dim=1)
        if self.diagnostic_mode:
            print(f"Combined features:     [{combined.min():.3f}, {combined.max():.3f}] shape={combined.shape}")
        
        # Final classification - track each layer in fusion
        x = combined
        for i, layer in enumerate(self.fusion_classifier):
            x = layer(x)
            if self.diagnostic_mode:
                layer_name = layer.__class__.__name__
                if hasattr(x, 'shape'):
                    print(f"Fusion layer {i} ({layer_name:12s}): [{x.min():.3f}, {x.max():.3f}] shape={x.shape}")
        
        logits = x
        if self.diagnostic_mode:
            print(f"Final logits:          [{logits.min():.3f}, {logits.max():.3f}]")
            print("="*70)
            print()
        
        return logits


# ============================================================
# HYBRID DATASET (Returns Spatial + DCT)
# ============================================================

class HybridSpatialDCTDataset(StegoDCTSpatialDataset):
    """
    Extended dataset that returns BOTH spatial pixels AND DCT coefficients.
    
    This is needed for the hybrid model that processes both domains:
        - Spatial: RGB pixels for FilterBank HPF processing
        - DCT: JPEG DCT coefficients for frequency analysis
    
    Returns:
        spatial_img: [3, H_pixels, W_pixels] - RGB pixels normalized to [0, 1]
        dct_blocks: [C, H_blocks, W_blocks, 8, 8] - DCT coefficients
        label: 0 (clean) or 1 (stego)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Target pixel size (should match target_blocks × 8)
        self.target_pixels = (self.target_blocks[0] * 8, self.target_blocks[1] * 8)
        logger.info("HybridDataset: target_pixels=%s", self.target_pixels)
    
    def _load_spatial_image(self, img_path):
        """
        Load and preprocess spatial image (RGB pixels).

        Takes a random 256x256 crop of image to save compute
        
        Args:
            img_path: Path to image file
        
        Returns:
            Tensor of shape [3, H, W] normalized to [0, 1]
        """
        try:
            # Increase PIL decompression bomb limit for large images
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None  # Disable limit
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Resize/crop to target size
            w, h = img.size
            target_h, target_w = self.target_pixels
            
            # Ensure image is large enough to crop from
            if w < target_w or h < target_h:
                # If image is smaller than target, pad it
                new_img = Image.new('RGB', (target_w, target_h), (0, 0, 0))
                paste_x = (target_w - w) // 2
                paste_y = (target_h - h) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img
                w, h = img.size

            # Cropping logic
            if self.train:
                # Random crop for training
                left = np.random.randint(0, w - target_w + 1)
                top = np.random.randint(0, h - target_h + 1)
            else:
                # Center crop for validation/testing
                left = (w - target_w) // 2
                top = (h - target_h) // 2
            
            img = img.crop((left, top, left + target_w, top + target_h))
            
            # Convert to tensor and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            
            return img_tensor
            
        except Exception as e:
            logger.error("Failed to load/crop spatial image %s: %s", img_path, e)
            return torch.zeros((3, *self.target_pixels), dtype=torch.float32)
    
    def __getitem__(self, idx):
        """
        Returns:
            spatial_img: [3, H_pixels, W_pixels] - RGB pixels normalized to [0, 1]
            dct_blocks: [C, H_blocks, W_blocks, 8, 8] - DCT coefficients
            label: 0 (clean) or 1 (stego)
        """
        # Get DCT features from parent class
        dct_blocks, label = super().__getitem__(idx)
        
        # Load spatial pixels
        row = self.records.iloc[idx]
        img_path = Path(row[self.path_col])
        if not img_path.is_absolute():
            img_path = (self.img_root / img_path).resolve()
        
        spatial_img = self._load_spatial_image(img_path)
        
        # Apply same augmentation to both spatial and DCT
        if self.train:
            import random
            
            # Horizontal flip
            if random.random() < 0.5:
                spatial_img = torch.flip(spatial_img, dims=[2])  # Flip width
                dct_blocks = torch.flip(dct_blocks, dims=[2])
            
            # Vertical flip
            if random.random() < 0.5:
                spatial_img = torch.flip(spatial_img, dims=[1])  # Flip height
                dct_blocks = torch.flip(dct_blocks, dims=[1])
        
        return spatial_img, dct_blocks, label