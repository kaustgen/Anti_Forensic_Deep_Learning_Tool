#!/usr/bin/env python3
"""Small smoke test: run one epoch of training on a small subset to validate end-to-end pipeline.
Uses CPU and num_workers=0 to be safe.
"""
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from sten_dct import StegoImageDataset
from sten_dct_train import StegoDetectionCNN, StegoDetectionTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    EXCEL = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'
    IMG_ROOT = Path('.')

    logging.info('Building dataset...')
    ds = StegoImageDataset(EXCEL, IMG_ROOT, dct_channels=['Y', 'Cb', 'Cr'])

    N = min(32, len(ds))
    indices = list(range(N))
    sub = Subset(ds, indices)

    # Split small subset
    train_size = int(0.75 * N)
    val_size = N - train_size
    train_ds, val_ds = torch.utils.data.random_split(sub, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    BATCH_SIZE = 8
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    features, _ = ds[0]
    input_dim = features.shape[0]
    logging.info('Feature dimension: %d', input_dim)

    model = StegoDetectionCNN(input_dim=input_dim, hidden_dims=[128, 64], dropout=0.3)
    trainer = StegoDetectionTrainer(model=model, device='cpu', learning_rate=1e-3)

    logging.info('Starting 1-epoch smoke training...')
    trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=1, early_stop_patience=2)
    logging.info('Smoke training finished')


if __name__ == '__main__':
    main()
