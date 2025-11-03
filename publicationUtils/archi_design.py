"""
ARCHITECTURE DIAGRAM SPECIFICATION

This guide explains exactly what to draw in your publication diagram.
Use draw.io, PowerPoint, or TikZ (LaTeX) to create the visual.
"""

# ============================================================
# MAIN ARCHITECTURE OVERVIEW
# ============================================================

ARCHITECTURE = {
    "input": {
        "jpeg_image": "Single JPEG file (e.g., 256×256 pixels)",
        "splits_into": ["spatial_pixels", "dct_coefficients"]
    },
    
    # ========== LEFT BRANCH: SPATIAL ==========
    "spatial_branch": {
        "input": {
            "name": "Spatial Pixels",
            "shape": "[B, 3, 256, 256]",
            "description": "RGB pixel values (decompressed JPEG)"
        },
        
        "preprocessing": {
            "layer": "High-Pass Filter Bank",
            "type": "15 fixed filters (not trainable)",
            "filters": [
                "1× KV (Square 5×5)",
                "4× Edge (left, right, top, bottom)",
                "1× KP (point filter)",
                "1× KB (edge detection)",
                "8× 2D Gabor (4 orientations × 2 phases)"
            ],
            "output_shape": "[B, 45, 256, 256]",
            "purpose": "Extract noise residuals for stego detection"
        },
        
        "type1_layers": {
            "name": "Type 1: Noise Extraction",
            "layers": [
                {
                    "layer_num": 1,
                    "operation": "Conv2d(45→64, 3×3) + BN + TLU(T=3)",
                    "shape": "[B, 64, 256, 256]",
                    "pooling": "NO"
                },
                {
                    "layer_num": 2,
                    "operation": "Conv2d(64→16, 3×3) + BN + TLU(T=3)",
                    "shape": "[B, 16, 256, 256]",
                    "pooling": "NO"
                }
            ],
            "purpose": "Extract high-frequency noise patterns"
        },
        
        "type2_layers": {
            "name": "Type 2: Unpooled Residuals",
            "layers": [
                {
                    "layer_num": 3,
                    "operation": "Conv2d(16→16, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 16, 256, 256]",
                    "pooling": "NO",
                    "residual": "x + identity"
                },
                {
                    "layer_num": 4,
                    "operation": "Conv2d(16→16, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 16, 256, 256]",
                    "pooling": "NO",
                    "residual": "x + identity"
                },
                {
                    "layer_num": 5,
                    "operation": "Conv2d(16→16, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 16, 256, 256]",
                    "pooling": "NO",
                    "residual": "x + identity"
                },
                {
                    "layer_num": 6,
                    "operation": "Conv2d(16→16, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 16, 256, 256]",
                    "pooling": "NO",
                    "residual": "x + identity"
                },
                {
                    "layer_num": 7,
                    "operation": "Conv2d(16→16, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 16, 256, 256]",
                    "pooling": "NO",
                    "residual": "x + identity"
                }
            ],
            "purpose": "Preserve spatial detail (NO pooling for 5 layers)"
        },
        
        "type3_layers": {
            "name": "Type 3: Pooled Residuals",
            "layers": [
                {
                    "layer_num": 8,
                    "operation": "Conv2d(16→64, 3×3)×2 + BN + ReLU + AvgPool(3×3, stride=2)",
                    "shape": "[B, 64, 128, 128]",
                    "pooling": "YES (256→128)"
                },
                {
                    "layer_num": 9,
                    "operation": "Conv2d(64→64, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 64, 128, 128]",
                    "pooling": "NO",
                    "residual": "x + identity"
                },
                {
                    "layer_num": 10,
                    "operation": "Conv2d(64→256, 3×3)×2 + BN + ReLU + AvgPool(3×3, stride=2)",
                    "shape": "[B, 256, 64, 64]",
                    "pooling": "YES (128→64)"
                },
                {
                    "layer_num": 11,
                    "operation": "Conv2d(256→256, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 256, 64, 64]",
                    "pooling": "NO",
                    "residual": "x + identity"
                }
            ],
            "purpose": "Hierarchical feature extraction"
        },
        
        "output": {
            "operation": "AdaptiveAvgPool2d(1×1)",
            "shape": "[B, 256]",
            "description": "Spatial feature vector"
        }
    },
    
    # ========== RIGHT BRANCH: FREQUENCY ==========
    "frequency_branch": {
        "input": {
            "name": "DCT Coefficients",
            "shape": "[B, 3, 32, 32, 8, 8]",
            "description": "JPEG DCT blocks (Y+Cb+Cr channels)"
        },
        
        "preprocessing": {
            "operation": "Reshape: [B, 3, 32, 32, 8, 8] → [B, 192, 32, 32]",
            "description": "Flatten 64 DCT coeffs into channels (3×64=192)"
        },
        
        "type1_layers": {
            "name": "Type 1: Frequency Extraction",
            "layers": [
                {
                    "layer_num": 1,
                    "operation": "Conv2d(192→64, 3×3) + BN + ReLU",
                    "shape": "[B, 64, 32, 32]",
                    "pooling": "NO"
                },
                {
                    "layer_num": 2,
                    "operation": "Conv2d(64→16, 3×3) + BN + ReLU",
                    "shape": "[B, 16, 32, 32]",
                    "pooling": "NO"
                }
            ],
            "purpose": "Extract DCT patterns"
        },
        
        "type2_layers": {
            "name": "Type 2: Unpooled Residuals",
            "layers": [
                {
                    "layer_num": 3,
                    "operation": "Conv2d(16→16, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 16, 32, 32]",
                    "pooling": "NO",
                    "residual": "x + identity"
                },
                # Layers 4-7: Same as layer 3 (16→16, no pooling)
                "... (layers 4-7 identical to spatial branch)",
            ],
            "purpose": "Preserve DCT block structure"
        },
        
        "type3_layers": {
            "name": "Type 3: Pooled Residuals",
            "layers": [
                {
                    "layer_num": 8,
                    "operation": "Conv2d(16→64, 3×3)×2 + BN + ReLU + AvgPool(3×3, stride=2)",
                    "shape": "[B, 64, 16, 16]",
                    "pooling": "YES (32→16)"
                },
                {
                    "layer_num": 9,
                    "operation": "Conv2d(64→64, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 64, 16, 16]",
                    "pooling": "NO"
                },
                {
                    "layer_num": 10,
                    "operation": "Conv2d(64→128, 3×3)×2 + BN + ReLU + AvgPool(3×3, stride=2)",
                    "shape": "[B, 128, 8, 8]",
                    "pooling": "YES (16→8)"
                },
                {
                    "layer_num": 11,
                    "operation": "Conv2d(128→128, 3×3)×2 + BN + ReLU + Residual",
                    "shape": "[B, 128, 8, 8]",
                    "pooling": "NO"
                }
            ],
            "purpose": "Hierarchical frequency features"
        },
        
        "output": {
            "operation": "AdaptiveAvgPool2d(1×1)",
            "shape": "[B, 128]",
            "description": "Frequency feature vector"
        }
    },
    
    # ========== FUSION ==========
    "fusion": {
        "input": {
            "operation": "Concatenate([spatial, frequency])",
            "shape": "[B, 384]",  # 256 + 128
            "description": "Combined feature vector"
        },
        
        "classifier": {
            "layers": [
                {
                    "operation": "Linear(384→256) + BN + ReLU + Dropout(0.3)",
                    "shape": "[B, 256]"
                },
                {
                    "operation": "Linear(256→128) + BN + ReLU + Dropout(0.3)",
                    "shape": "[B, 128]"
                },
                {
                    "operation": "Linear(128→2)",
                    "shape": "[B, 2]",
                    "description": "Logits for [Clean, Stego]"
                }
            ]
        },
        
        "output": {
            "operation": "Softmax",
            "shape": "[B, 2]",
            "description": "Probabilities: P(Clean), P(Stego)"
        }
    }
}


# ============================================================
# SIMPLIFIED DIAGRAM LABELS (For Drawing)
# ============================================================

DIAGRAM_LABELS = """
┌─────────────────────────────────────────────────────────────┐
│                      JPEG Image Input                        │
│                     (256×256 pixels)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
┌────────▼────────┐          ┌────────▼─────────┐
│ SPATIAL BRANCH  │          │ FREQUENCY BRANCH │
│                 │          │                  │
│ Pixels          │          │ DCT Coefficients │
│ [3,256,256]     │          │ [3,32,32,8,8]    │
└────────┬────────┘          └────────┬─────────┘
         │                            │
┌────────▼────────┐          ┌────────▼─────────┐
│ HPF Filterbank  │          │ Reshape to       │
│ (15 filters)    │          │ [192,32,32]      │
│ → [45,256,256]  │          │                  │
└────────┬────────┘          └────────┬─────────┘
         │                            │
┌────────▼────────┐          ┌────────▼─────────┐
│ Type 1 Layers   │          │ Type 1 Layers    │
│ (2 layers)      │          │ (2 layers)       │
│ → [16,256,256]  │          │ → [16,32,32]     │
└────────┬────────┘          └────────┬─────────┘
         │                            │
┌────────▼────────┐          ┌────────▼─────────┐
│ Type 2 Layers   │          │ Type 2 Layers    │
│ (5 unpooled)    │          │ (5 unpooled)     │
│ + Residuals     │          │ + Residuals      │
│ → [16,256,256]  │          │ → [16,32,32]     │
└────────┬────────┘          └────────┬─────────┘
         │                            │
┌────────▼────────┐          ┌────────▼─────────┐
│ Type 3 Layers   │          │ Type 3 Layers    │
│ (4 pooled)      │          │ (4 pooled)       │
│ + Residuals     │          │ + Residuals      │
│ → [256,64,64]   │          │ → [128,8,8]      │
└────────┬────────┘          └────────┬─────────┘
         │                            │
┌────────▼────────┐          ┌────────▼─────────┐
│ Global Avg Pool │          │ Global Avg Pool  │
│ → [256]         │          │ → [128]          │
└────────┬────────┘          └────────┬─────────┘
         │                            │
         └─────────────┬──────────────┘
                       │
              ┌────────▼────────┐
              │ Concatenate     │
              │ [384]           │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Fusion Layers   │
              │ (3 FC layers)   │
              │ 384→256→128→2   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Softmax         │
              │ [Clean, Stego]  │
              └─────────────────┘
"""

print(DIAGRAM_LABELS)