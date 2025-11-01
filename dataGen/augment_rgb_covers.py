#!/usr/bin/env python3
"""
Augment RGB cover images to create training dataset diversity.

Takes images from clean_images/ and creates augmented variants to expand
the dataset from ~300 images to 2000+ images while preserving image quality
suitable for steganographic embedding.

Author: Kaleb Austgen
Date: October 30, 2025
"""

import sys
from pathlib import Path
import logging
from tqdm import tqdm
import shutil
import math

# Add deprecated folder to path to import image_augment
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from image_augment import ImageAugmentor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

# Input: Your diverse RGB cover images
INPUT_DIR = BASE_DIR / 'clean_images'

# Output: Augmented cover images (ready for stego embedding)
OUTPUT_DIR = BASE_DIR / 'rgb_augmented_covers'

# Target total number of augmented images
TARGET_TOTAL_IMAGES = 2400

# Use mild augmentation to preserve stego embedding quality
# Mild = small rotations, gentle blur, conservative transforms
USE_MILD = True

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================
# MAIN AUGMENTATION PIPELINE
# ============================================================

def augment_image(augmentor, img_path, output_dir, num_variants):
    """
    Create augmented variants of a single image.
    
    Args:
        augmentor: ImageAugmentor instance
        img_path: Path to input image
        output_dir: Directory to save variants
        num_variants: Number of variants to create
        
    Returns:
        Number of variants successfully created
    """
    created_count = 0
    
    # Variant 0: Original (copy as-is)
    original_out = output_dir / f"{img_path.stem}_aug0{img_path.suffix}"
    if not original_out.exists():
        try:
            shutil.copy2(img_path, original_out)
            created_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {img_path.name}: {e}")
            return created_count
    
    # Variants 1 to N: Augmented versions
    for variant_idx in range(1, num_variants):
        output_name = f"{img_path.stem}_aug{variant_idx}{img_path.suffix}"
        output_path = output_dir / output_name
        
        if not output_path.exists():
            try:
                success = augmentor.augment_and_save(
                    input_path=img_path,
                    output_path=output_path,
                    transforms='random',
                    mild=USE_MILD
                )
                if success:
                    created_count += 1
            except Exception as e:
                logger.error(f"Failed to augment {img_path.name} (variant {variant_idx}): {e}")
    
    return created_count


def main():
    """Generate augmented RGB cover images."""
    
    logger.info("="*70)
    logger.info("RGB IMAGE AUGMENTATION AND GENERATION")
    logger.info("="*70)
    
    # Find all input images (jpg, jpeg, png, webp)
    input_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        input_images.extend(INPUT_DIR.glob(ext))
    
    input_images = sorted(input_images)
    
    if not input_images:
        logger.error(f"No images found in {INPUT_DIR}")
        logger.error("Please check the INPUT_DIR path.")
        return
    
    # Calculate variants per image based on target total
    
    variants_per_image = math.ceil(TARGET_TOTAL_IMAGES / len(input_images))
    expected_total = len(input_images) * variants_per_image
    
    logger.info(f"Input directory:      {INPUT_DIR}")
    logger.info(f"Output directory:     {OUTPUT_DIR}")
    logger.info(f"Input images found:   {len(input_images)}")
    logger.info(f"Target total images:  {TARGET_TOTAL_IMAGES}")
    logger.info(f"Variants per image:   {variants_per_image}")
    logger.info(f"Expected total:       {expected_total}")
    logger.info(f"Augmentation mode:    {'MILD' if USE_MILD else 'AGGRESSIVE'}")
    logger.info(f"Random seed:          {RANDOM_SEED}")
    logger.info("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize augmentor
    augmentor = ImageAugmentor(seed=RANDOM_SEED)
    
    # Process all images
    total_created = 0
    failed_images = []
    
    logger.info("\nStarting augmentation...")
    for img_path in tqdm(input_images, desc="Augmenting images", unit="img"):
        try:
            created = augment_image(augmentor, img_path, OUTPUT_DIR, variants_per_image)
            total_created += created
            
            if created < variants_per_image:
                failed_images.append((img_path.name, created, variants_per_image))
        except Exception as e:
            logger.error(f"Unexpected error processing {img_path.name}: {e}")
            failed_images.append((img_path.name, 0, variants_per_image))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Input images:         {len(input_images)}")
    logger.info(f"Created variants:     {total_created}")
    logger.info(f"Expected variants:    {expected_total}")
    logger.info(f"Success rate:         {100 * total_created / expected_total:.1f}%")
    logger.info(f"Output directory:     {OUTPUT_DIR}")
    
    if failed_images:
        logger.warning(f"\n{len(failed_images)} images had issues:")
        for img_name, created, expected in failed_images[:10]:  # Show first 10
            logger.warning(f"  {img_name}: {created}/{expected} variants created")
        if len(failed_images) > 10:
            logger.warning(f"  ... and {len(failed_images) - 10} more")
    else:
        logger.info("\nAll images augmented successfully!")
    
    logger.info("="*70)
    logger.info("\nNext step: Run steg_rgb_j_uniward.py to embed stego in these covers")
    logger.info("="*70)


if __name__ == '__main__':
    main()
