# Add this to calculate_payload_stats.py and run on BOSS dataset
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).parent.parent
excel_path = base_dir / 'BOSS_stego_training.xlsx'
df = pd.read_excel(excel_path)
payloads = df['Payload (bpp AC DCT)'].dropna()

print(f"BOSS Payload Stats:")
print(f"  Mean: {payloads.mean():.4f}")
print(f"  Median: {payloads.median():.4f}")
print(f"  Min: {payloads.min():.4f}")
print(f"  Max: {payloads.max():.4f}")

# Compare to your old dataset
df_old = pd.read_excel('dataGen/stego_training.xlsx')
payloads_old = df_old['Payload (bpp AC DCT)'].dropna()

print(f"\nOld Dataset Payload Stats:")
print(f"  Mean: {payloads_old.mean():.4f}")
print(f"  Median: {payloads_old.median():.4f}")