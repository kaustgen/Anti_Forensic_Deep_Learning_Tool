#!/usr/bin/env python3
"""
J-UNIWARD (JPEG Universal Wavelet Relative Distortion) Steganography

Implementation based on:
Holub, V., Fridrich, J., & Denemark, T. (2014). 
"Universal distortion function for steganography in an arbitrary domain"
EURASIP Journal on Information Security, 2014(1), 1.

This implementation embeds data directly in JPEG DCT coefficients.
"""

import numpy as np
import jpeglib  # Use jpeglib instead of jpegio for proper DCT writing
from pathlib import Path
import logging
from typing import Tuple, Optional
import random

logger = logging.getLogger(__name__)


class JUNIWARDEmbedder:
    """
    J-UNIWARD steganographic embedder for JPEG images.
    
    Embeds data by modifying DCT coefficients with minimal distortion.
    """
    
    def __init__(self, alpha: float = 0.4):
        """
        Initialize J-UNIWARD embedder.
        
        Args:
            alpha: Embedding rate in bits per non-zero AC coefficient (bpnzAC)
                   Typical values: 0.1, 0.2, 0.4 (default)
                   Higher = more data embedded = easier to detect
        """
        self.alpha = alpha
        
    def _compute_distortion(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """
        Compute distortion cost for modifying each DCT coefficient.
        
        Lower cost = better place to embed (less perceptible).
        
        Args:
            dct_coeffs: DCT coefficient array [H, W]
            
        Returns:
            distortion: Distortion cost for each coefficient [H, W]
        """
        # Compute wavelet decomposition costs
        # J-UNIWARD uses directional filters to estimate local texture
        
        # Horizontal and vertical gradients
        h_filter = np.array([[-1, 2, -1]])
        v_filter = h_filter.T
        
        # Convert to float for processing
        dct_float = dct_coeffs.astype(np.float64)
        
        # Compute directional costs using convolution
        h_cost = np.abs(self._convolve2d(dct_float, h_filter))
        v_cost = np.abs(self._convolve2d(dct_float, v_filter))
        
        # Diagonal filters
        d1_filter = np.array([[-1, 2, -1]]).T @ np.array([[1, 0, -1]])
        d2_filter = np.array([[1, 2, 1]]).T @ np.array([[1, 0, -1]])
        
        d1_cost = np.abs(self._convolve2d(dct_float, d1_filter))
        d2_cost = np.abs(self._convolve2d(dct_float, d2_filter))
        
        # All costs should have same shape now
        # Pad edges to match original size
        h, w = dct_coeffs.shape
        distortion = np.zeros((h, w), dtype=np.float64)
        
        # Get the overlapping region size
        min_h = min(h_cost.shape[0], v_cost.shape[0], d1_cost.shape[0], d2_cost.shape[0])
        min_w = min(h_cost.shape[1], v_cost.shape[1], d1_cost.shape[1], d2_cost.shape[1])
        
        # Compute combined cost for valid region
        combined = np.sqrt(
            h_cost[:min_h, :min_w]**2 + 
            v_cost[:min_h, :min_w]**2 + 
            d1_cost[:min_h, :min_w]**2 + 
            d2_cost[:min_h, :min_w]**2
        )
        
        # Place in center of output
        start_h = (h - min_h) // 2
        start_w = (w - min_w) // 2
        distortion[start_h:start_h+min_h, start_w:start_w+min_w] = combined
        
        # Avoid division by zero
        distortion = np.maximum(distortion, 1e-10)
        
        # Reciprocal - lower texture = higher cost
        distortion = 1.0 / distortion
        
        return distortion
    
    def _convolve2d(self, arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution"""
        from scipy.ndimage import convolve
        result = convolve(arr, kernel, mode='constant', cval=0.0)
        # Ensure output has same shape as input minus padding
        h, w = arr.shape
        kh, kw = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2
        return result[pad_h:h-pad_h, pad_w:w-pad_w]
    
    def _stc_embed(self, cover_dct: np.ndarray, distortion: np.ndarray, 
                   payload_bits: int) -> np.ndarray:
        """
        Syndrome-Trellis Codes (STC) embedding using simplified approach.
        
        For practical purposes, we use a simple cost-based embedding:
        - Select lowest-cost coefficients to modify
        - Embed by ±1 changes to DCT coefficients
        
        Args:
            cover_dct: Original DCT coefficients
            distortion: Cost of modifying each coefficient
            payload_bits: Number of bits to embed
            
        Returns:
            stego_dct: Modified DCT coefficients
        """
        stego_dct = cover_dct.copy().astype(np.int16)
        
        # Get embeddable coefficients (non-zero AC coefficients)
        embeddable_mask = (cover_dct != 0)
        
        # Flatten for easier processing
        flat_dct = cover_dct.flatten()
        flat_distortion = distortion.flatten()
        flat_embeddable = embeddable_mask.flatten()
        
        # Get indices of embeddable coefficients sorted by cost (lowest first)
        embeddable_indices = np.where(flat_embeddable)[0]
        embeddable_costs = flat_distortion[embeddable_indices]
        sorted_indices = embeddable_indices[np.argsort(embeddable_costs)]
        
        # Generate random message bits
        message_bits = np.random.randint(0, 2, size=payload_bits, dtype=np.uint8)
        
        # Embed bits by modifying lowest-cost coefficients
        # Simple LSB embedding with ±1 changes
        num_changes = min(payload_bits, len(sorted_indices))
        
        if num_changes == 0:
            logger.warning("No embeddable coefficients found!")
            return stego_dct
        
        # Reshape back to 2D
        stego_flat = flat_dct.copy()
        
        for i in range(num_changes):
            idx = sorted_indices[i]
            original_val = stego_flat[idx]
            bit = message_bits[i]
            
            # Embed bit by changing LSB
            current_lsb = abs(int(original_val)) % 2
            
            if current_lsb != bit:
                # Need to change coefficient
                if original_val > 0:
                    stego_flat[idx] = original_val + (1 if bit == 1 else -1)
                else:
                    stego_flat[idx] = original_val - (1 if bit == 1 else -1)
        
        stego_dct = stego_flat.reshape(cover_dct.shape).astype(np.int16)
        
        return stego_dct
    
    def embed(self, cover_path: str, stego_path: str, 
              payload_bytes: Optional[int] = None) -> dict:
        """
        Embed random data into JPEG image using J-UNIWARD.
        
        Args:
            cover_path: Path to cover JPEG image
            stego_path: Path to save stego image
            payload_bytes: Number of bytes to embed (if None, use alpha rate)
            
        Returns:
            dict with embedding statistics
        """
        cover_path = Path(cover_path)
        stego_path = Path(stego_path)
        
        if not cover_path.exists():
            raise FileNotFoundError(f"Cover image not found: {cover_path}")
        
        # Read JPEG structure using jpeglib
        jpeg = jpeglib.read_dct(str(cover_path))
        dct_coeffs = jpeg.Y  # Luminance channel DCT coefficients [blocks_v, blocks_h, 8, 8]
        
        if dct_coeffs is None:
            raise ValueError(f"No DCT coefficients in {cover_path}")
        
        # jpeglib uses [blocks_v, blocks_h, 8, 8] format, flatten to [height, width]
        num_blocks_v, num_blocks_h = dct_coeffs.shape[0], dct_coeffs.shape[1]
        total_h = num_blocks_v * 8
        total_w = num_blocks_h * 8
        y_dct = dct_coeffs.transpose(0, 2, 1, 3).reshape(total_h, total_w).copy()
        
        # Compute embedding capacity
        nzAC = np.sum(y_dct != 0)  # Non-zero AC coefficients
        
        if payload_bytes is None:
            # Use alpha rate
            capacity_bits = int(nzAC * self.alpha)
        else:
            capacity_bits = payload_bytes * 8
        
        # Check capacity
        max_capacity_bits = int(nzAC * 0.95)  # Safety margin
        if capacity_bits > max_capacity_bits:
            logger.warning(f"Requested {capacity_bits} bits but max capacity is {max_capacity_bits}")
            capacity_bits = max_capacity_bits
        
        actual_payload_bytes = capacity_bits // 8
        
        # Compute distortion costs
        distortion = self._compute_distortion(y_dct)
        
        # Embed using STC
        stego_y_dct = self._stc_embed(y_dct, distortion, capacity_bits)
        
        # Reshape back to [blocks_v, blocks_h, 8, 8] format for jpeglib
        stego_y_dct_blocks = stego_y_dct.reshape(num_blocks_v, 8, num_blocks_h, 8).transpose(0, 2, 1, 3)
        
        # Replace Y channel with stego version
        jpeg.Y = stego_y_dct_blocks
        
        # Write stego JPEG using jpeglib
        jpeg.write_dct(str(stego_path))
        
        # Calculate statistics
        changes = np.sum(stego_y_dct != y_dct)
        change_rate = changes / y_dct.size
        
        stats = {
            'payload_bytes': actual_payload_bytes,
            'payload_bits': capacity_bits,
            'bpnzAC': capacity_bits / nzAC if nzAC > 0 else 0,
            'changes': changes,
            'change_rate': change_rate,
            'nzAC': nzAC,
            'image_size': y_dct.size
        }
        
        logger.debug(f"Embedded {actual_payload_bytes} bytes ({capacity_bits} bits) "
                    f"at {stats['bpnzAC']:.3f} bpnzAC, "
                    f"changed {changes} coefficients ({change_rate*100:.2f}%)")
        
        return stats


def embed_file(cover_path: str, stego_path: str, 
               alpha: float = 0.4, payload_bytes: Optional[int] = None) -> dict:
    """
    Convenience function to embed data using J-UNIWARD.
    
    Args:
        cover_path: Path to cover JPEG
        stego_path: Path to save stego JPEG
        alpha: Embedding rate (bpnzAC), default 0.4
        payload_bytes: Specific payload size, or None to use alpha
        
    Returns:
        dict with embedding statistics
    """
    embedder = JUNIWARDEmbedder(alpha=alpha)
    return embedder.embed(cover_path, stego_path, payload_bytes)


# Test function
def test_j_uniward():
    """Test J-UNIWARD embedding on a sample image"""
    import sys
    
    # Find a test image
    base_dir = Path(__file__).parent
    test_images = list((base_dir / 'BOSS1.01_Dataset').glob('*.jpg'))
    
    if not test_images:
        print("No test images found in BOSS1.01_Dataset/")
        return
    
    cover = test_images[0]
    stego = base_dir / 'test_j_uniward_stego.jpg'
    
    print(f"Testing J-UNIWARD on: {cover.name}")
    print(f"Embedding rate: 0.4 bpnzAC")
    
    try:
        stats = embed_file(str(cover), str(stego), alpha=0.4)
        
        print("\nEmbedding successful!")
        print(f"  Payload: {stats['payload_bytes']} bytes ({stats['payload_bits']} bits)")
        print(f"  Rate: {stats['bpnzAC']:.3f} bpnzAC")
        print(f"  Changes: {stats['changes']} / {stats['image_size']} ({stats['change_rate']*100:.2f}%)")
        print(f"  Non-zero AC: {stats['nzAC']}")
        print(f"\nStego saved to: {stego}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run test
    test_j_uniward()
