# Author: Kaleb Austgen
# Date: 10/20/25
# Purpose: Extract DCT-domain features from JPEGs (histograms + co-occurrence)
# and provide a Dataset that yields feature vectors suitable for classifier training.

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import sys
import logging
from tqdm import tqdm

# jpegio can read quantized DCT coefficients directly from JPEGs. Import
# it conditionally because some environments don't have it or it may crash on
# incompatible binaries. We'll fall back to a decoded-image DCT computation.
try:
    import jpegio as jio
    HAVE_JPEGLIB = True
except Exception:
    jio = None
    HAVE_JPEGLIB = False

# Some jpegio builds can crash the Python process when reading certain
# images. For safety during analysis and debugging we keep an explicit
# flag that disables in-process jpegio usage; set to True only if you
# are confident jpegio is stable in this environment.
USE_JPEGLIB_IN_PROCESS = True
from hashlib import sha1
import subprocess
import json

# Cache directory for child-produced coef arrays
JPEGLIB_CACHE = Path(__file__).parent.parent / 'dataGen' / '.jpegio_cache'
JPEGLIB_CACHE.mkdir(parents=True, exist_ok=True)

# module logger
logger = logging.getLogger(__name__)

# Loading and Processing images
class StegoImageDataset(Dataset):
    def __init__(self, excel_path, img_root: Path, img_size=128, dct_channels=None, cache_dir=None):
        """
        Args:
            excel_path: path to Excel file with columns [image_path, is_stego]
                       First column: JPEG file paths
                       Second column: True/False (True = steganography with Steghide)
            records: Alternative - pandas DataFrame with columns [path, label]
            img_root: base path for image files (if paths in Excel are relative)
            img_size: target image size
            bitplane: whether to include LSB bitplane channel
            use_dct: whether to extract DCT coefficients instead of pixel values
            dct_channels: which color channels to use for DCT ['Y', 'Cb', 'Cr']
            dct_features: 'histogram', 'cooccurrence', or 'both'
        """
        # Load from Excel (no header row) to ensure first column is path and second is label
        self.records = pd.read_excel(excel_path, header=None)
        logger.info("Loaded %d records from %s", len(self.records), excel_path)
        logger.debug("Columns: %s", list(self.records.columns))

        # Verify we have at least 2 columns (path, label)
        if len(self.records.columns) < 2:
            raise ValueError(f"Excel file must have at least 2 columns (path, label). Found: {list(self.records.columns)}")

        # Column names
        self.path_col = self.records.columns[0]
        self.label_col = self.records.columns[1]
        
        # Convert True/False strings to boolean if needed
        if self.records[self.label_col].dtype == 'object':
            self.records[self.label_col] = self.records[self.label_col].map({
                'True': True, 'TRUE': True, True: True,
                'False': False, 'FALSE': False, False: False
            })
        
        # Display dataset label counts (coerce to boolean series first)
        try:
            labels_bool = self.records[self.label_col].astype(bool)
            logger.info("Dataset breakdown: Clean=%d Steghide=%d", (~labels_bool).sum(), labels_bool.sum())
        except Exception:
            # If label coercion fails, print raw value counts
            logger.info("Dataset breakdown (raw value counts):\n%s", self.records[self.label_col].value_counts())

        self.img_root = Path(img_root) if img_root else Path('.')
        self.img_size = img_size
        self.dct_channels = dct_channels or ['Y', 'Cb', 'Cr']

        logger.debug("set img_root=%s img_size=%d dct_channels=%s", self.img_root, self.img_size, self.dct_channels)

        # Transform used only for optional image-based fallbacks
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        logger.debug("created base_transform")

        # Precompute 8x8 DCT basis for numpy fallback
        logger.debug("about to create dct basis")
        self._dct_basis_8 = self._create_dct_basis(8)
        logger.debug("created dct basis")

        # Cache directory for pre-computed features. Use an absolute directory
        default_cache = Path(__file__).parent / '.feature_cache'
        self.cache_dir = (Path(cache_dir) if cache_dir else default_cache).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # We will lazily load features from disk when requested
        logger.info("Using lazy on-disk feature cache: %s", self.cache_dir)
    
    def precompute_cache(self, start=0, end=None):
        """
        Precompute and write missing feature cache files to disk.

        This method writes per-image cache files (sha1-based names) but does
        not retain features in memory, so it's safe for large datasets.
        """
        n = len(self.records)
        if end is None or end > n:
            end = n

        for idx in tqdm(range(start, end), desc="Precomputing feature cache"):
            row = self.records.iloc[idx]
            img_path = Path(row[self.path_col])
            if not img_path.is_absolute():
                img_path = (self.img_root / img_path).resolve()

            cache_file = self._cache_path_for(img_path)
            if cache_file.exists():
                # Check mtime validity
                try:
                    img_mtime = img_path.stat().st_mtime
                    cache_mtime = cache_file.stat().st_mtime
                    if cache_mtime >= img_mtime:
                        continue
                except Exception:
                    # If stat fails, just recompute
                    pass

            try:
                img = self._load_image(img_path)
                features = self._extract_dct_features(img_path, img)
                # atomic write
                tmp = cache_file.with_suffix('.pt.tmp')
                torch.save(features, tmp)
                tmp.replace(cache_file)
            except Exception as e:
                logger.exception("Failed to precompute cache for %s: %s", img_path, e)

    def __len__(self):
        return len(self.records)

    def _load_image(self, p: Path):
        img = Image.open(p).convert('RGB')
        return img
    
    def _bitplane_channel(self, img: Image.Image):
        g = np.asarray(img.convert('L'), dtype=np.uint8)
        lsb = (g & 1).astype(np.float32)
        return lsb
    
    def _create_dct_basis(self, n):
        # DCT-II orthonormal basis for nxn blocks
        k = np.arange(n)
        i = k.reshape((n, 1))
        alpha = np.sqrt(2.0 / n) * np.ones(n)
        alpha[0] = np.sqrt(1.0 / n)
        basis = alpha * np.cos(np.pi * (2 * i + 1) * k / (2.0 * n))
        return basis

    def _dct2(self, block):
        B = self._dct_basis_8
        return B.dot(block).dot(B.T)
    
    def _dct_coefficients(self, img_path: Path):
        """
        Extract quantized DCT coefficients directly from JPEG file.
        This preserves the original DCT domain information crucial for stego detection.
        Returns coefficients for each 8x8 block and color channel.
        """
        # Prefer jpegio if available (reads quantized coefficients). If it
        # fails for any reason we fall back to computing block DCTs from the
        # decompressed image (not identical to quantized JPEG coefficients
        # but suitable as a fallback).
        # If jpegio is available, prefer to use the subprocess-based worker
        # to avoid crashing the main process. The worker writes a compressed
        # .npz file we can load safely.
        if HAVE_JPEGLIB and USE_JPEGLIB_IN_PROCESS:
            # child-based read with caching
            h = sha1(str(img_path).encode('utf-8')).hexdigest()
            out_npz = JPEGLIB_CACHE / f"{h}.npz"
            if not out_npz.exists():
                worker = Path(__file__).parent / 'jpegio_worker.py'
                cmd = [sys.executable, str(worker), str(img_path), str(out_npz)]
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                except Exception as e:
                    logger.error("jpegio worker failed to start for %s: %s", img_path, e)
                    proc = None

                if proc is None or proc.returncode != 0:
                    rc = proc.returncode if proc is not None else None
                    logger.warning("jpegio worker failed (rc=%s) for %s", rc, img_path)
                    if proc is not None:
                        logger.debug('jpegio worker stdout: %s', proc.stdout)
                        logger.debug('jpegio worker stderr: %s', proc.stderr)
                    logger.info('Falling back to decoded DCT for %s', img_path)
                else:
                    # successful child; load the npz
                    try:
                        data = np.load(str(out_npz))
                        coef_arrays = [data[k] for k in sorted(data.files)]
                        return coef_arrays
                    except Exception as e:
                        logger.warning("Failed to load coef npz %s: %s", out_npz, e)
                        logger.info('Falling back to decoded DCT for %s', img_path)

            else:
                try:
                    data = np.load(str(out_npz))
                    coef_arrays = [data[k] for k in sorted(data.files)]
                    return coef_arrays
                except Exception as e:
                    logger.warning("Failed to load cached coef npz %s: %s", out_npz, e)
                    logger.info('Falling back to decoded DCT for %s', img_path)

        # Fallback: compute DCTs from decoded YCbCr image
        img_ycbcr = Image.open(img_path).convert('YCbCr')
        channels = img_ycbcr.split()
        coef_arrays = []
        for c in channels:
            arr = np.asarray(c, dtype=np.float32)
            h, w = arr.shape
            pad_h = (8 - (h % 8)) % 8
            pad_w = (8 - (w % 8)) % 8
            if pad_h or pad_w:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
            H, W = arr.shape
            coefs = np.zeros_like(arr)
            for by in range(0, H, 8):
                for bx in range(0, W, 8):
                    block = arr[by:by+8, bx:bx+8].astype(np.float32) - 128.0
                    d = self._dct2(block)
                    coefs[by:by+8, bx:bx+8] = np.round(d)
            coef_arrays.append(coefs.astype(np.int32))
        return coef_arrays

    def _cache_path_for(self, img_path: Path) -> Path:
        """Return the per-image cache file path for a given image path."""
        h = sha1(str(img_path).encode('utf-8')).hexdigest()
        return self.cache_dir / f"{h}.pt"
    
    def _extract_dct_histogram(self, coef_arrays, num_bins=64, freq_bands=None):
        """
        Create histogram features from DCT coefficients

        Args:
            coef_arrays: List of DCT coefficient arrays per channel
            num_bins: number of histogram bins
            freq_bands: list of (low, mid, high) frequency maks or None for all coefficients
        """
        histograms = []
        for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
            if channel_name not in self.dct_channels or channel_idx >= len(coef_arrays):
                continue

            coeffs = coef_arrays[channel_idx].flatten()

            # create histogram with range based on typical jpeg dct values
            hist, _ = np.histogram(coeffs, bins=num_bins, range=(-1024, 1024))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-8)  # normalize
            histograms.append(hist)

        return np.concatenate(histograms) if histograms else np.array([], dtype=np.float32)
    
    def _extract_dct_cooccurence(self, coef_arrays, levels=32):
        """
        Create co-occurence matrix features from DCT coefficients
        Analyzes spatial relationships between adjacent DCT coefficients
        """
        features = []

        for channel_idx, channel_name in enumerate(['Y', 'Cb', 'Cr']):
            if channel_name not in self.dct_channels or channel_idx >= len(coef_arrays):
                continue

            coeffs = coef_arrays[channel_idx]

            # Quantize coefficients to reduce matrix size
            coeffs_quantized = np.clip(coeffs, -512, 511)
            coeffs_quantized = ((coeffs_quantized + 512) * levels / 1024.0).astype(np.int32)
            coeffs_quantized = np.clip(coeffs_quantized, 0, levels - 1)

            # Compute horizontal and vertical co-occurrence
            h, v = coeffs_quantized.shape
            co_oc_h = np.zeros((levels, levels), dtype=np.float32)
            co_oc_v = np.zeros((levels, levels), dtype=np.float32)

            # Horizontal pairs
            for i in range(h):
                for j in range(v - 1):
                    a = coeffs_quantized[i, j]
                    b = coeffs_quantized[i, j + 1]
                    co_oc_h[a, b] += 1

            # Vertical pairs
            for i in range(h - 1):
                for j in range(v):
                    a = coeffs_quantized[i, j]
                    b = coeffs_quantized[i + 1, j]
                    co_oc_v[a, b] += 1

            # Normalize
            co_oc_h = co_oc_h / (co_oc_h.sum() + 1e-8)
            co_oc_v = co_oc_v / (co_oc_v.sum() + 1e-8)

            # Extract texture features (contrast, energy, homogeneity)
            features.extend(self._cooccurence_features(co_oc_h))
            features.extend(self._cooccurence_features(co_oc_v))

        return np.array(features, dtype=np.float32)
    
    def _cooccurence_features(self, co_oc_matrix):
        """Extract statistical features from co-occurence matrix"""

        contrast = 0
        energy = 0 # uniformity
        homogeneity = 0

        n = co_oc_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                contrast += (i - j) ** 2 * co_oc_matrix[i, j]
                energy += co_oc_matrix[i, j] ** 2
                homogeneity += co_oc_matrix[i, j] / (1 + abs(i - j))
        
        return [contrast, energy, homogeneity]
    
    def _extract_dct_features(self, img_path: Path, img: Image.Image):
        """Main method to extract DCT features"""

        # Use jpegio for coefficient extraction
        coef_arrays = self._dct_coefficients(img_path)

        hist_features = self._extract_dct_histogram(coef_arrays)
        co_oc_features = self._extract_dct_cooccurence(coef_arrays)

        parts = []
        if hist_features.size:
            parts.append(hist_features)
        if co_oc_features.size:
            parts.append(co_oc_features)
        if not parts:
            return torch.tensor([], dtype=torch.float32)

        features = np.concatenate(parts)
        return torch.tensor(features, dtype=torch.float32)
    
    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        label = bool(row[self.label_col])

        img_path = Path(row[self.path_col])
        if not img_path.is_absolute():
            img_path = (self.img_root / img_path).resolve()

        cache_file = self._cache_path_for(img_path)

        # If cache exists and is newer than image, load it. Otherwise compute
        # and write a new cache.
        load_from_cache = False
        if cache_file.exists():
            try:
                img_mtime = img_path.stat().st_mtime
                cache_mtime = cache_file.stat().st_mtime
                if cache_mtime >= img_mtime:
                    load_from_cache = True
            except Exception:
                # If stat fails, be conservative and try to load
                load_from_cache = True

        if load_from_cache:
            try:
                features = torch.load(cache_file)
            except Exception:
                logger.exception("Failed to load cache %s, recomputing", cache_file)
                img = self._load_image(img_path)
                features = self._extract_dct_features(img_path, img)
                # atomic write
                tmp = cache_file.with_suffix('.pt.tmp')
                torch.save(features, tmp)
                tmp.replace(cache_file)
        else:
            try:
                img = self._load_image(img_path)
                features = self._extract_dct_features(img_path, img)
                # write cache
                tmp = cache_file.with_suffix('.pt.tmp')
                torch.save(features, tmp)
                tmp.replace(cache_file)
            except Exception:
                logger.exception("Failed to compute features for %s", img_path)
                features = torch.tensor([], dtype=torch.float32)

        return features, torch.tensor(1 if label else 0, dtype=torch.long)

#EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'

    def create_dataset(self, excel_path=Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'):
        """Function that returns the dataset"""
        dataset_full = StegoImageDataset(
            excel_path=excel_path,
            img_root=Path('.')
        )
        return dataset_full

# if __name__ == "__main__":

#     dataset_full = StegoImageDataset(
#         excel_path=EXCEL_PATH,
#         img_root=Path('.')
#     )
#     # Example 3: Test loading a sample
#     if len(dataset_full) > 0:
#         features, label = dataset_full[0]
#         logger.info("Feature vector shape: %s", features.shape)
#         logger.info("Label (1=Steghide, 0=Clean): %s", label)

#     # Create a DataLoader for potential downstream tests (use num_workers=0 when debugging)
#     train_loader = DataLoader(
#         dataset_full,
#         batch_size=32,
#         shuffle=True,
#         num_workers=4
#     )

#     logger.info("Dataset ready with %d images", len(dataset_full))
#     logger.info("Feature dimensions: %s", dataset_full[0][0].shape)

#     # Quick guarded per-file DCT check to help reproduce native crashes safely.
#     logger.info("Running guarded per-file DCT coefficient checks (first 4 files)")
#     for i in range(min(4, len(dataset_full))):
#         try:
#             row = dataset_full.records.iloc[i]
#             p = Path(row[dataset_full.path_col])
#             if not p.is_absolute():
#                 p = (dataset_full.img_root / p).resolve()
#             logger.debug("Checking file %d: %s", i, p)
#             try:
#                 coefs = dataset_full._dct_coefficients(p)
#                 logger.info("  got %d coef arrays; shapes: %s", len(coefs), [c.shape for c in coefs])
#             except Exception as e:
#                 logger.exception("  _dct_coefficients raised for %s: %s", p, e)
#         except Exception as e:
#             logger.exception("  outer failure on row %d: %s", i, e)