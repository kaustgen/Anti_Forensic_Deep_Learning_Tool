"""sten_train.py

This file contains a small image classification pipeline (PyTorch) used to detect
steganography-modified JPEGs. The high-level flow is:

1. Read an Excel spreadsheet where each row contains: (image_path, label)
    - The first column should be a path to a .jpg file (absolute or relative).
    - The second column should be a boolean-like label (True/False, 1/0, yes/no).
2. Build a Dataset that loads the image, resizes it, converts to tensor, and
    optionally appends a lightweight LSB bitplane channel (useful for LSB stego).
3. Create a small convolutional neural network (SmallCNN) and train it.
4. Save the best model (by validation accuracy) to disk.

Configuration is provided as top-level globals near the bottom of this file
(EXCEL_PATH, IMG_ROOT, IMG_SIZE, USE_BITPLANE, EPOCHS, etc.). Edit those
variables and then run the module (e.g. `python3 -m Stegnography.sten_train`).

Dependencies: pandas, openpyxl, pillow, torch, torchvision, scikit-learn
"""


"""
Modern Function
Extract quantized DCT coefficients from the JPEG
Compute hisogram and co-occurence features for certain frequency bands
Compare them statistically against clean images
Use Random Forest trained on these features to estimate probability of stegnography
"""

from pathlib import Path
# train_test_split is used to split the DataFrame into train/validation sets
from sklearn.model_selection import train_test_split

# Image handling and tabular IO
from PIL import Image
import pandas as pd
import numpy as np

# PyTorch imports for model, training loop and data loading
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os


class StegoImageDataset(Dataset):
    def __init__(self, records, img_root: Path, img_size=128, bitplane=False):
        # `records` is a pandas DataFrame containing at least two columns: [path, label]
        self.records = records
        # base path used to resolve relative image paths from the excel sheet
        self.img_root = Path(img_root)
        self.img_size = img_size
        # whether to compute and append a single-channel LSB bitplane
        self.bitplane = bitplane

        # base image pre-processing: resize to a square and convert to a float tensor
        # resulting tensors are in the range [0.0, 1.0] with shape (C, H, W)
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def _load_image(self, p: Path):
        # Open the file with PIL and convert to RGB (3 channels). This function
        # intentionally raises if the file does not exist or is unreadable so the
        # caller can see the failure immediately.
        img = Image.open(p).convert('RGB')
        return img

    def _bitplane_channel(self, img: Image.Image):
        # Compute the least-significant-bit (LSB) plane of the image. We first
        # convert to grayscale and then extract the LSB of each pixel (0 or 1).
        # This produces a lightweight single-channel signal which is often
        # useful for detecting LSB-based steganography.
        g = np.asarray(img.convert('L'), dtype=np.uint8)
        lsb = (g & 1).astype(np.float32)  # map pixels to 0/1
        return lsb

    def _dct_coefficients(self, img: Image.Image):
        pass

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_path = Path(row[0])
        label = bool(row[1])

        # if path is relative, join with root
        if not img_path.is_absolute():
            img_path = (self.img_root / img_path).resolve()

        img = self._load_image(img_path)
        img_t = self.base_transform(img)  # C x H x W, floats 0..1

        if self.bitplane:
            bp = self._bitplane_channel(img)
            # resize bp to img_size
            bp_img = Image.fromarray((bp * 255).astype(np.uint8))
            bp_t = T.ToTensor()(bp_img.resize((self.img_size, self.img_size)))
            # the bitplane is single-channel; concatenate as an extra channel
            img_t = torch.cat([img_t, bp_t], dim=0)

        # Return: image tensor and integer class label (0 or 1)
        return img_t, torch.tensor(1 if label else 0, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # Forward pass: extract convolutional features then run the small MLP
        # classifier head to produce two logits (class 0 and class 1).
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for xb, yb in loader:
        # Move data to device (CPU or GPU)
        xb = xb.to(device)
        yb = yb.to(device)

        opt.zero_grad()
        logits = model(xb)               # forward
        loss = loss_fn(logits, yb)       # compute loss
        loss.backward()                  # backprop
        opt.step()                       # update weights

        # accumulate loss and accuracy stats for logging
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    # return average loss and accuracy over the epoch
    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            # same as training loop but without gradient updates
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

    return total_loss / total, correct / total


# ----------------------
# Configuration (edit these globals then run the module)
# ----------------------
# Path to the Excel file (two columns: path, label)
EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'

# Root directory to resolve relative image paths in the Excel sheet
IMG_ROOT = Path(__file__).parent.parent

# Image size used by the CNN (square)
IMG_SIZE = 128

# If True, add a single-channel LSB bitplane concatenated to RGB channels
USE_BITPLANE = True

# Training params
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Output model file
OUT_MODEL = Path(__file__).parent / 'sten_model.pt'


def main():
    df = pd.read_excel(EXCEL_PATH, header=None)
    # assume two columns: path, label
    if df.shape[1] < 2:
        raise SystemExit('Excel must have at least two columns: path and label')

    # convert labels to booleans (or 0/1)
    df[1] = df[1].apply(lambda v: True if str(v).lower() in ('true', '1', 'yes') else False)

    # Split the DataFrame into a training and validation set while preserving
    # the class distribution using stratified sampling. This provides a small
    # validation holdout to monitor generalization during training.
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[1], random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_channels = 3 + (1 if USE_BITPLANE else 0)
    train_ds = StegoImageDataset(train_df.reset_index(drop=True), img_root=IMG_ROOT, img_size=IMG_SIZE, bitplane=USE_BITPLANE)
    val_ds = StegoImageDataset(val_df.reset_index(drop=True), img_root=IMG_ROOT, img_size=IMG_SIZE, bitplane=USE_BITPLANE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SmallCNN(in_channels=in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {ep}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_MODEL)
            print(f"Saved best model (val_acc={best_val_acc:.4f}) to {OUT_MODEL}")


if __name__ == '__main__':
    main()
