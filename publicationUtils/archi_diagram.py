# Author: Kaleb Austgen
# Date: 11/1/25
# Purpose: Assist in building an architecture diagram of the complicated CNN model

"""
Grad-CAM Visualization for Hybrid Spatial-Frequency CNN

Generates Class Activation Maps showing which spatial regions the model
focuses on when detecting steganography. Highlights areas where stego
artifacts are most prominent.

"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'Stegnography'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import pandas as pd

from sten_dct_cnn import HybridSpatialDCTDataset, SpatialFrequencyHybridCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Shows which spatial regions contribute most to model's decision.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Your hybrid CNN model
            target_layer: Layer to visualize (usually last conv before pooling)
                         For spatial branch: model.spatial_layer11
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to capture forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, spatial_img, dct_blocks, target_class):
        """
        Generate Class Activation Map.
        
        Args:
            spatial_img: [1, 3, 256, 256] spatial input
            dct_blocks: [1, 3, 32, 32, 8, 8] DCT input
            target_class: 0 (clean) or 1 (stego)
        
        Returns:
            cam: [H, W] heatmap (values in [0, 1])
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(spatial_img, dct_blocks)
        
        # Get score for target class
        score = logits[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Compute weights (global average pooling of gradients)
        # gradients: [1, C, H, W]
        # activations: [1, C, H, W]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def apply_colormap(cam, cmap='jet'):
    """
    Apply colormap to CAM.
    
    Args:
        cam: [H, W] grayscale heatmap
        cmap: Colormap name ('jet', 'hot', 'viridis')
    
    Returns:
        [H, W, 3] RGB heatmap
    """
    colormap = plt.get_cmap(cmap)
    cam_color = colormap(cam)[:, :, :3]  # Drop alpha channel
    return (cam_color * 255).astype(np.uint8)


def overlay_cam_on_image(img, cam, alpha=0.5):
    """
    Overlay CAM heatmap on original image.
    
    Args:
        img: [3, H, W] tensor (normalized to [0, 1])
        cam: [H_cam, W_cam] heatmap (values in [0, 1])
        alpha: Transparency of overlay (0 = invisible, 1 = opaque)
    
    Returns:
        [H, W, 3] RGB image with overlay
    """
    # Convert image to numpy [H, W, 3]
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Resize CAM to match image size
    H, W = img_np.shape[:2]
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H)))
    cam_resized = cam_resized.astype(np.float32) / 255.0
    
    # Apply colormap
    cam_color = apply_colormap(cam_resized, cmap='jet')
    
    # Blend
    overlay = (alpha * cam_color + (1 - alpha) * img_np).astype(np.uint8)
    
    return overlay


def find_payload_samples(excel_path, payload_rates=[0.1, 0.2, 0.3, 0.4], tolerance=0.04):
    """
    Find one clean and one stego sample for each payload rate.
    
    Args:
        excel_path: Path to Excel metadata
        payload_rates: List of target payload rates
        tolerance: Tolerance for payload matching
    
    Returns:
        List of (index, label, payload) tuples
    """
    df = pd.read_excel(excel_path)
    
    # Column names from steg_boss_j_uniward.py:
    # "File Path", "Stegnography Applied?", "Payload Category", "Payload Size (bytes)",
    # "Payload (bpp AC DCT)", "Payload (bytes)", "Payload (bits)", "Image Dimensions", 
    # "Non-zero AC DCT", "RGB"
    
    selected_indices = []
    
    for payload in payload_rates:
        # Find clean image (Stegnography Applied? = False)
        clean_df = df[df['Stegnography Applied?'] == False]
        if len(clean_df) > 0:
            clean_idx = clean_df.sample(1).index[0]
            selected_indices.append((clean_idx, 0, 0.0, payload))
            logger.info(f"Payload {payload:.1f}: Selected clean image at index {clean_idx}")
        
        # Find stego image with target payload
        # Payload is in column "Payload (bpp AC DCT)"
        stego_df = df[df['Stegnography Applied?'] == True].copy()
        
        # Convert payload column to float (handle 'N/A' strings)
        stego_df['Payload (bpp AC DCT)'] = pd.to_numeric(stego_df['Payload (bpp AC DCT)'], errors='coerce')
        
        # Filter by payload rate
        stego_df = stego_df[
            (stego_df['Payload (bpp AC DCT)'] >= payload - tolerance) & 
            (stego_df['Payload (bpp AC DCT)'] <= payload + tolerance)
        ]
        
        if len(stego_df) > 0:
            stego_idx = stego_df.sample(1).index[0]
            actual_payload = stego_df.loc[stego_idx, 'Payload (bpp AC DCT)']
            selected_indices.append((stego_idx, 1, actual_payload, payload))
            logger.info(f"Payload {payload:.1f}: Selected stego image at index {stego_idx} (actual: {actual_payload:.4f})")
        else:
            logger.warning(f"No stego images found for payload {payload:.1f} ± {tolerance}")
    
    return selected_indices


def visualize_grad_cam(model, dataset, sample_list, output_dir, device='cuda'):
    """
    Generate Grad-CAM visualizations for selected images.
    
    Args:
        model: Trained hybrid CNN
        dataset: HybridSpatialDCTDataset
        sample_list: List of (index, label, actual_payload, target_payload) tuples
        output_dir: Where to save visualizations
        device: 'cuda' or 'cpu'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up Grad-CAM (target: last spatial conv layer)
    grad_cam = GradCAM(model, target_layer=model.spatial_layer11)
    
    model.eval()
    model.to(device)
    
    for idx, true_label, actual_payload, target_payload in sample_list:
        # Load sample
        spatial_img, dct_blocks, label = dataset[idx]
        
        # Add batch dimension
        spatial_img = spatial_img.unsqueeze(0).to(device)
        dct_blocks = dct_blocks.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(spatial_img, dct_blocks)
            pred_probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = pred_probs[0, pred_class].item()
        
        # Generate CAM for predicted class
        cam = grad_cam.generate_cam(spatial_img, dct_blocks, target_class=pred_class)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img_display = spatial_img[0].cpu()
        axes[0].imshow(img_display.permute(1, 2, 0))
        true_class = "Stego" if true_label else "Clean"
        payload_str = f" ({actual_payload:.4f} bpnzAC)" if true_label else ""
        axes[0].set_title(f'Original Image\nTrue: {true_class}{payload_str}')
        axes[0].axis('off')
        
        # Heatmap only
        axes[1].imshow(cam, cmap='jet')
        pred_class_str = "Stego" if pred_class else "Clean"
        axes[1].set_title(f'Grad-CAM Heatmap\nPred: {pred_class_str} ({confidence:.2%})')
        axes[1].axis('off')
        
        # Overlay
        overlay = overlay_cam_on_image(img_display, cam, alpha=0.4)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save with descriptive filename
        class_str = "stego" if true_label else "clean"
        payload_str = f"payload{actual_payload:.3f}" if true_label else "payload0.000"
        pred_str = "correct" if pred_class == true_label else "incorrect"
        filename = f'gradcam_{class_str}_{payload_str}_target{target_payload:.1f}_{pred_str}_idx{idx}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filename}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Generate Grad-CAM visualizations for publication."""
    
    # ========== CONFIGURATION ==========
    MODEL_PATH = Path('Stegnography/models_hybrid_lowrate/grey_only/best_hybrid_model.pth')
    EXCEL_PATH = Path('dataGen/BOSS_stego_metadata.xlsx')
    IMG_ROOT = Path('dataGen')
    OUTPUT_DIR = Path('publicationUtils/grad_cam_outputs')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Payload rates to visualize (one clean + one stego per rate)
    PAYLOAD_RATES = [0.1, 0.2, 0.3, 0.4]
    TOLERANCE = 0.04  # ±0.04 bpnzAC tolerance for matching
    
    logger.info("="*70)
    logger.info("GRAD-CAM VISUALIZATION GENERATION")
    logger.info("="*70)
    logger.info("Model: %s", MODEL_PATH)
    logger.info("Dataset: %s", EXCEL_PATH)
    logger.info("Device: %s", DEVICE)
    logger.info("Payload rates: %s", PAYLOAD_RATES)
    logger.info("="*70)
    
    # ========== FIND SAMPLES ==========
    logger.info("\nFinding samples for each payload rate...")
    sample_list = find_payload_samples(EXCEL_PATH, PAYLOAD_RATES, TOLERANCE)
    logger.info(f"Selected {len(sample_list)} samples total")
    
    # ========== LOAD MODEL ==========
    logger.info("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    model = SpatialFrequencyHybridCNN(
        num_channels=3,
        target_blocks=(32, 32),
        dropout=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    logger.info("Model loaded from epoch %d (val_acc: %.2f%%)",
                checkpoint.get('epoch', -1),
                checkpoint.get('val_acc', 0) * 100)
    
    # ========== LOAD DATASET ==========
    logger.info("\nLoading dataset...")
    dataset = HybridSpatialDCTDataset(
        excel_path=EXCEL_PATH,
        img_root=IMG_ROOT,
        dct_channels=['Y', 'Cb', 'Cr'],
        target_blocks=(32, 32),
        train=False  # No augmentation
    )
    logger.info("Dataset loaded: %d images", len(dataset))
    
    # ========== GENERATE VISUALIZATIONS ==========
    logger.info("\nGenerating Grad-CAM visualizations...")
    visualize_grad_cam(
        model=model,
        dataset=dataset,
        sample_list=sample_list,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE!")
    logger.info("="*70)
    logger.info("Visualizations saved to: %s", OUTPUT_DIR.resolve())
    logger.info("Total images: %d", len(sample_list))
    logger.info("  - Clean images: %d", sum(1 for _, label, _, _ in sample_list if label == 0))
    logger.info("  - Stego images: %d", sum(1 for _, label, _, _ in sample_list if label == 1))


if __name__ == '__main__':
    main()