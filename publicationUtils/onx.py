#!/usr/bin/env python3
"""
Export model to TorchScript format (more compatible than ONNX).
Can be visualized in Netron just like ONNX.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'Stegnography'))

import torch
from sten_dct_cnn import SpatialFrequencyHybridCNN

def export_to_torchscript():
    """Export model to TorchScript format."""
    
    # Paths
    MODEL_PATH = Path('Stegnography/models_hybrid_lowrate/grey_only/best_hybrid_model.pth')
    OUTPUT_PATH = Path('publicationUtils/hybrid_model_architecture.pt')
    
    print("="*70)
    print("EXPORTING MODEL TO TORCHSCRIPT FORMAT")
    print("="*70)
    print(f"Input:  {MODEL_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    model = SpatialFrequencyHybridCNN(
        num_channels=3,
        target_blocks=(32, 32),
        dropout=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2%})")
    
    # Create example inputs
    print("\nCreating example inputs...")
    spatial_input = torch.randn(1, 3, 256, 256)
    dct_input = torch.randn(1, 3, 32, 32, 8, 8)
    
    # Trace the model (creates TorchScript)
    print("\nTracing model with TorchScript...")
    try:
        traced_model = torch.jit.trace(model, (spatial_input, dct_input))
        
        # Save
        traced_model.save(str(OUTPUT_PATH))
        
        print(f"\n✓ Successfully exported to: {OUTPUT_PATH.resolve()}")
        print("\nNext steps:")
        print("  1. Open Netron: https://netron.app/")
        print(f"  2. Upload: {OUTPUT_PATH.name}")
        print("  3. Explore the architecture!")
        print("\nAlternative (command line):")
        print(f"  netron {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        print("\nThis might be due to dynamic control flow in the model.")
        print("Try the torchinfo summary instead (see below).")
    
    print("="*70)

if __name__ == '__main__':
    export_to_torchscript()