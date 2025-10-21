"""
Quick smoke test for sten_train environment.

This script performs a minimal check:
- loads the Excel configured in `sten_train.EXCEL_PATH`
- builds a dataset with 2 samples
- instantiates the CNN with the configured bitplane option
- runs a single forward pass on CPU and prints shapes

Run this before full training to verify your venv and dependencies.
"""

from pathlib import Path
import sys

# Ensure the repository root is on sys.path so `Stegnography` can be imported when
# running this script directly.
proj_root = Path(__file__).parent.parent.resolve()
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

try:
    import Stegnography.sten_train as st
except Exception as e:
    print(f"Failed to import sten_train: {e}")
    raise

import pandas as pd
import torch


def run_smoke():
    print("Excel path:", st.EXCEL_PATH)
    if not Path(st.EXCEL_PATH).exists():
        print("ERROR: Excel file not found at EXCEL_PATH. Update the path in sten_train.py and try again.")
        return 2

    df = pd.read_excel(st.EXCEL_PATH, header=None)
    if df.shape[0] < 2:
        print("ERROR: Excel must contain at least 2 rows for smoke test.")
        return 2

    # use just first two rows
    sample_df = df.iloc[:2].reset_index(drop=True)
    ds = st.StegoImageDataset(sample_df, img_root=st.IMG_ROOT, img_size=st.IMG_SIZE, bitplane=st.USE_BITPLANE)
    xb, yb = ds[0]
    xb2, yb2 = ds[1]
    batch = torch.stack([xb, xb2], dim=0)
    print("Input batch shape:", batch.shape)

    device = torch.device('cpu')
    model = st.SmallCNN(in_channels=3 + (1 if st.USE_BITPLANE else 0)).to(device)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    print("Output shape:", out.shape)
    print("Smoke test succeeded — model forward pass completed.")
    return 0


if __name__ == '__main__':
    try:
        rc = run_smoke()
        sys.exit(rc)
    except Exception as e:
        print("Smoke test failed with exception:", e)
        raise
