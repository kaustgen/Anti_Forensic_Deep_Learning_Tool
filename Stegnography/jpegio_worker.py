#!/usr/bin/env python3
"""Worker to safely read JPEG quantized DCT coefficients using jpegio.

Usage: python jpegio_worker.py /path/to/image.jpg /path/to/out.npz

Exits:
  0 on success (writes .npz with arrays 'c0','c1','c2')
  2 if jpegio import failed
  3 if jpegio.read raised an exception
"""
import sys
from pathlib import Path
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("usage: jpegio_worker.py INPUT_JPEG OUTPUT_NPZ", file=sys.stderr)
        return 4

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    try:
        import jpegio as jio
    except Exception as e:
        print("jpegio import failed:", e, file=sys.stderr)
        return 2

    try:
        j = jio.read(str(inp))
    except Exception as e:
        print("jpegio.read failed:", repr(e), file=sys.stderr)
        return 3

    # j.coef_arrays is typically a list of numpy arrays per channel
    arrays = getattr(j, 'coef_arrays', None)
    if arrays is None:
        print("jpegio returned no coef_arrays", file=sys.stderr)
        return 5

    # Save arrays to npz with deterministic keys
    to_save = {}
    for i, a in enumerate(arrays):
        # ensure numpy array
        to_save[f'c{i}'] = np.asarray(a)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(out, **to_save)
    except Exception as e:
        print("failed to write npz:", repr(e), file=sys.stderr)
        return 6

    return 0

if __name__ == '__main__':
    sys.exit(main())
