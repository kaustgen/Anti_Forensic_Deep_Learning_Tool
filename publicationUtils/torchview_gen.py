#!/usr/bin/env python3
"""
Generate publication-quality architecture visualization using torchview.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'Stegnography'))

import torch
from torchview import draw_graph
from sten_dct_cnn import SpatialFrequencyHybridCNN

def visualize_architecture():
    """Create architecture visualization."""
    
    print("="*70)
    print("GENERATING ARCHITECTURE VISUALIZATION")
    print("="*70)
    
    # Create model
    model = SpatialFrequencyHybridCNN(
        num_channels=3,
        target_blocks=(32, 32),
        dropout=0.3
    )
    model.eval()
    
    # Create dummy inputs
    spatial_input = torch.randn(1, 3, 256, 256)
    dct_input = torch.randn(1, 3, 32, 32, 8, 8)
    
    # Generate graph
    print("\nGenerating computational graph...")
    model_graph = draw_graph(
        model,
        input_data=(spatial_input, dct_input),
        expand_nested=True,
        depth=4,  # Show 4 levels of nesting
        device='cpu',
        graph_name='Hybrid Spatial-Frequency CNN',
        save_graph=True,
        directory='publicationUtils',
        filename='architecture_graph'
    )
    
    print("\n✓ Saved visualizations:")
    print("  - publicationUtils/architecture_graph.png")
    print("  - publicationUtils/architecture_graph.gv (Graphviz source)")
    print("="*70)

if __name__ == '__main__':
    visualize_architecture()