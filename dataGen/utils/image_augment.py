#!/usr/bin/env python3
# Author: Kaleb Austgen
# Date: 10/21/25
# Purpose: 
"""

Provides various image augmentation techniques to create more diverse training data:
- Rotation (small angles to preserve stego data)
- Gaussian blur (fuzziness)
- Noise injection (Gaussian, salt & pepper)
- Brightness/contrast adjustment
- JPEG compression quality variation
- Slight scaling/cropping

These augmentations help the CNN generalize better by exposing it to realistic variations
it might encounter in the wild.
"""

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageAugmentor:
    """Apply various augmentations to images for training data diversity."""
    
    def __init__(self, seed=None):
        """
        Initialize augmentor.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def rotate(self, img, angle=None, angle_range=(-5, 5)):
        """
        Rotate image by a small angle. Small angles preserve stego data better.
        
        Args:
            img: PIL Image
            angle: Specific angle in degrees (if None, random from angle_range)
            angle_range: Tuple of (min, max) degrees for random rotation
            
        Returns:
            Rotated PIL Image
        """
        if angle is None:
            angle = random.uniform(*angle_range)
        return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    
    def gaussian_blur(self, img, radius=None, radius_range=(0.5, 2.0)):
        """
        Apply Gaussian blur (fuzziness).
        
        Args:
            img: PIL Image
            radius: Blur radius (if None, random from radius_range)
            radius_range: Tuple of (min, max) blur radius
            
        Returns:
            Blurred PIL Image
        """
        if radius is None:
            radius = random.uniform(*radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius))
    
    def add_gaussian_noise(self, img, mean=0, std=None, std_range=(5, 15)):
        """
        Add Gaussian noise to image.
        
        Args:
            img: PIL Image
            mean: Mean of noise distribution
            std: Standard deviation (if None, random from std_range)
            std_range: Tuple of (min, max) std values
            
        Returns:
            Noisy PIL Image
        """
        if std is None:
            std = random.uniform(*std_range)
        
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(mean, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def add_salt_pepper_noise(self, img, amount=None, amount_range=(0.001, 0.01)):
        """
        Add salt and pepper noise (random black/white pixels).
        
        Args:
            img: PIL Image
            amount: Fraction of pixels to corrupt (if None, random from amount_range)
            amount_range: Tuple of (min, max) corruption fraction
            
        Returns:
            Noisy PIL Image
        """
        if amount is None:
            amount = random.uniform(*amount_range)
        
        img_array = np.array(img)
        num_pixels = img_array.size
        num_corrupt = int(num_pixels * amount)
        
        # Salt (white pixels)
        coords = [np.random.randint(0, i - 1, num_corrupt // 2) for i in img_array.shape]
        img_array[coords[0], coords[1]] = 255
        
        # Pepper (black pixels)
        coords = [np.random.randint(0, i - 1, num_corrupt // 2) for i in img_array.shape]
        img_array[coords[0], coords[1]] = 0
        
        return Image.fromarray(img_array)
    
    def adjust_brightness(self, img, factor=None, factor_range=(0.7, 1.3)):
        """
        Adjust image brightness.
        
        Args:
            img: PIL Image
            factor: Brightness factor (1.0 = unchanged, <1 darker, >1 brighter)
            factor_range: Tuple of (min, max) brightness factors
            
        Returns:
            Brightness-adjusted PIL Image
        """
        if factor is None:
            factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, img, factor=None, factor_range=(0.7, 1.3)):
        """
        Adjust image contrast.
        
        Args:
            img: PIL Image
            factor: Contrast factor (1.0 = unchanged)
            factor_range: Tuple of (min, max) contrast factors
            
        Returns:
            Contrast-adjusted PIL Image
        """
        if factor is None:
            factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def adjust_sharpness(self, img, factor=None, factor_range=(0.5, 2.0)):
        """
        Adjust image sharpness.
        
        Args:
            img: PIL Image
            factor: Sharpness factor (1.0 = unchanged, <1 softer, >1 sharper)
            factor_range: Tuple of (min, max) sharpness factors
            
        Returns:
            Sharpness-adjusted PIL Image
        """
        if factor is None:
            factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
    def jpeg_compress(self, img, quality=None, quality_range=(75, 95)):
        """
        Re-compress image with different JPEG quality.
        
        Args:
            img: PIL Image
            quality: JPEG quality (1-100, if None random from quality_range)
            quality_range: Tuple of (min, max) quality values
            
        Returns:
            Re-compressed PIL Image
        """
        if quality is None:
            quality = random.randint(*quality_range)
        
        # Save to bytes and reload to simulate compression
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    def random_crop_and_resize(self, img, crop_factor=None, crop_factor_range=(0.8, 0.95)):
        """
        Randomly crop image and resize back to original size.
        
        Args:
            img: PIL Image
            crop_factor: Fraction of image to keep (if None, random from crop_factor_range)
            crop_factor_range: Tuple of (min, max) crop fractions
            
        Returns:
            Cropped and resized PIL Image
        """
        if crop_factor is None:
            crop_factor = random.uniform(*crop_factor_range)
        
        w, h = img.size
        new_w, new_h = int(w * crop_factor), int(h * crop_factor)
        
        # Random crop position
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.LANCZOS)
    
    def apply_random_augmentation(self, img, num_transforms=1, mild=True):
        """
        Apply random combination of augmentations.
        
        Args:
            img: PIL Image
            num_transforms: Number of random transforms to apply
            mild: If True, use conservative parameter ranges to preserve stego data
            
        Returns:
            Augmented PIL Image
        """
        # Available transformations
        transforms = [
            ('rotate', lambda: self.rotate(img, angle_range=(-3, 3) if mild else (-10, 10))),
            ('blur', lambda: self.gaussian_blur(img, radius_range=(0.3, 1.0) if mild else (0.5, 3.0))),
            ('gaussian_noise', lambda: self.add_gaussian_noise(img, std_range=(3, 8) if mild else (5, 20))),
            ('salt_pepper', lambda: self.add_salt_pepper_noise(img, amount_range=(0.001, 0.005) if mild else (0.001, 0.02))),
            ('brightness', lambda: self.adjust_brightness(img, factor_range=(0.85, 1.15) if mild else (0.6, 1.4))),
            ('contrast', lambda: self.adjust_contrast(img, factor_range=(0.85, 1.15) if mild else (0.6, 1.4))),
            ('sharpness', lambda: self.adjust_sharpness(img, factor_range=(0.7, 1.5) if mild else (0.3, 2.5))),
            ('jpeg_compress', lambda: self.jpeg_compress(img, quality_range=(85, 98) if mild else (70, 95))),
        ]
        
        # Randomly select transforms
        selected = random.sample(transforms, min(num_transforms, len(transforms)))
        
        result = img
        for name, transform_fn in selected:
            try:
                result = transform_fn()
            except Exception as e:
                logger.warning("Failed to apply %s: %s", name, e)
        
        return result
    
    def augment_and_save(self, input_path, output_path, transforms='random', mild=True):
        """
        Load image, apply augmentation, and save.
        
        Args:
            input_path: Path to input image
            output_path: Path to save augmented image
            transforms: 'random' or list of transform names
            mild: Use mild parameters to preserve stego data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            img = Image.open(input_path)
            
            if transforms == 'random':
                augmented = self.apply_random_augmentation(img, num_transforms=random.randint(1, 2), mild=mild)
            else:
                # Apply specific transforms
                augmented = img
                for transform_name in transforms:
                    if hasattr(self, transform_name):
                        augmented = getattr(self, transform_name)(augmented)
                    else:
                        logger.warning("Unknown transform: %s", transform_name)
            
            # Save with high quality to preserve as much data as possible
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format from extension
            ext = output_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                augmented.save(output_path, 'JPEG', quality=95)
            elif ext == '.png':
                augmented.save(output_path, 'PNG')
            else:
                augmented.save(output_path)
            
            return True
        except Exception as e:
            logger.error("Failed to augment %s: %s", input_path, e)
            return False


def create_augmented_variants(input_image, output_dir, num_variants=3, prefix='aug', mild=True):
    """
    Create multiple augmented variants of an image.
    
    Args:
        input_image: Path to input image
        output_dir: Directory to save variants
        num_variants: Number of variants to create
        prefix: Prefix for output filenames
        mild: Use mild augmentation parameters
        
    Returns:
        List of output paths
    """
    input_path = Path(input_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    augmentor = ImageAugmentor()
    output_paths = []
    
    for i in range(num_variants):
        output_name = f"{prefix}_{i}_{input_path.name}"
        output_path = output_dir / output_name
        
        if augmentor.augment_and_save(input_path, output_path, transforms='random', mild=mild):
            output_paths.append(output_path)
    
    return output_paths


if __name__ == '__main__':
    # Simple example: augment a single image
    logging.basicConfig(level=logging.INFO)
    
    # ============ CONFIGURATION ============
    INPUT_IMAGE = 'dataGen/cover_falls.jpg'  # Image to augment
    OUTPUT_DIR = 'augmented_images'           # Output directory
    NUM_VARIANTS = 3                          # Number of variants to create
    USE_MILD = True                           # Use mild augmentation
    # =======================================
    
    logger.info("Creating %d augmented variants of %s", NUM_VARIANTS, INPUT_IMAGE)
    outputs = create_augmented_variants(INPUT_IMAGE, OUTPUT_DIR, NUM_VARIANTS, mild=USE_MILD)
    logger.info("Created %d variants in %s", len(outputs), OUTPUT_DIR)
    for p in outputs:
        logger.info("  - %s", p)

