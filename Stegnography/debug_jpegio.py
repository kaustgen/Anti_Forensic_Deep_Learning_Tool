"""Per-file jpegio tester.

Runs jpegio.read on each image in a separate subprocess so that a native crash
in jpegio (if present) will not crash this parent process. Prints per-file
results.
"""
from pathlib import Path
import subprocess
import sys
import pandas as pd
import argparse

EXCEL = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'

def check_paths(limit=None):
    df = pd.read_excel(EXCEL, header=None)
    paths = df[0].tolist()
    if limit is not None:
        paths = paths[:limit]

    for p in paths:
        p = Path(p)
        print(f"Testing: {p}")
        if not p.exists():
            print("  -> File missing")
            continue

        # Run a small child that imports jpegio and attempts to read the file.
        # If jpegio segfaults, the child will exit with signal 11 and we'll
        # see non-zero returncode.
        child = [sys.executable, "-c", (
            "import sys\n"
            "try:\n"
            "    import jpegio as jio\n"
            "except Exception as e:\n"
            "    print('jpegio import failed:', e); sys.exit(2)\n"
            "try:\n"
            f"    j = jio.read('{str(p)}')\n"
            "    print('ok, coef arrays:', type(j.coef_arrays))\n"
            "except Exception as e:\n"
            "    print('jpegio.read raised:', repr(e)); sys.exit(3)\n"
        )]

        proc = subprocess.run(child, capture_output=True, text=True)
        print('  returncode:', proc.returncode)
        if proc.stdout:
            print('  stdout:', proc.stdout.strip())
        if proc.stderr:
            print('  stderr:', proc.stderr.strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=None, help='Number of files to test (default=all)')
    args = parser.parse_args()
    check_paths(limit=args.num)
