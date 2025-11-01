# Author: Kaleb Austgen
# Date: 10/20/25
# Purpose: Fully interconnected network to train 1D vectors based off of DCT coefficiencts from JPEGS to discover stenography

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import wandb
import weave
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
from sten_dct import StegoImageDataset
import logging
import os
# Directory where this script lives; used for saving/loading models and plots


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    Mix two samples with random weight to prevent overfitting and memorization.
    
    Args:
        x: Input batch
        y: Labels
        alpha: Beta distribution parameter (0.2 = moderate mixing)
        device: Device for tensors
    
    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


MODEL_DIR = Path(__file__).parent / 'training_data'

# Create training_data directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# module logger
logger = logging.getLogger(__name__)

# Default logging configuration for convenience when running as a script or
# interactive sessions. Users can override this configuration in their apps.
logging.basicConfig(level=logging.INFO)

class StegoDetectionCNN(nn.Module):
    """
    CNN for stegnography detection using DCT feature vectore and fully connected layers
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.5):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
        """
        super(StegoDetectionCNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build fully connected layers with BatchNorm and Dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Layer classification layer (binary: clean vs stego)
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance and hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class StegoDetectionTrainer:
    """Handles training, validation, and evaluation of the stego detection model."""

    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-4, class_weights=None):
        self.model = model.to(device)
        self.device = device

        # Loss functions

        # Use standard CrossEntropyLoss (FocalLoss causes mode collapse)
        self.criterion = nn.CrossEntropyLoss()

        # FocalLoss attempt (DISABLED - causes model to predict all same class)
        #self.criterion = FocalLoss(alpha=0.26, gamma=2.0)

        # Custom label smoothing cross entropy class
        #self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        # BCEWithLogitsLoss is for binary with single output - not appropriate for 2-logit output
        # self.criterion = nn.BCEWithLogitsLoss()

        # Select a loss function depending on if the classes are weighted or not...
        # if class_weights is not None:
        #     self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # else:
        #     self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Adam
        # self.optimizer = optim.Adam(
        #     model.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay
        # )

        # SGD optimizer with momentum and weight decay
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )

        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15,      # Restart every 15 epochs (increased from 10 for subtle patterns)
            T_mult=2,    # Double period after each restart
            eta_min=1e-6 # Minimum learning rate
        )

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 
        #     factor=0.5, 
        #     patience=3,
        #     min_lr=1e-6
        # )

        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs= []

    @weave.op()
    def train_epoch(self, train_loader, use_mixup=True, mixup_alpha=0.2):
        """Train for one epoch with optional mixup augmentation."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Apply mixup augmentation to prevent memorization
            if use_mixup and mixup_alpha > 0:
                features, labels_a, labels_b, lam = mixup_data(features, labels, alpha=mixup_alpha, device=self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                
                # Mixup loss: weighted combination of both labels
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients with SGD
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                # Statistics (use lambda-weighted accuracy)
                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # Weighted correct predictions
                correct += (lam * (predicted == labels_a).float() + (1 - lam) * (predicted == labels_b).float()).sum().item()
            else:
                # Standard training without mixup
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients with SGD
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                # Statistics
                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    @weave.op()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc='Validating'):
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # store for metrics
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_labels, all_predictions, all_probs

    def train(self, train_loader, val_loader, epochs=50, early_stop_patience=10, use_mixup=True, mixup_alpha=0.2):
        """Full training loop with early stopping and optional mixup augmentation."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch+1, epochs)
            
            # Train with mixup
            train_loss, train_acc = self.train_epoch(train_loader, use_mixup=use_mixup, mixup_alpha=mixup_alpha)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            logger.info("Train Loss: %.4f, Train Acc: %.2f%%", train_loss, train_acc)
            logger.info("Val Loss: %.4f, Val Acc: %.2f%%", val_loss, val_acc)

            # Call scheduler
            self.scheduler.step()

            # Learning rate scheduling
            #self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info("Learning Rate: %.6f", current_lr)

            # Log to wandb if available (guarded)
            if wandb is not None:
                try:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None
                    })
                except Exception:
                    logger.exception('wandb.log failed')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_path = MODEL_DIR / 'best_stego_model.pth'
                torch.save(self.model.state_dict(), str(best_path))
                logger.info("Saved best model to %s", best_path)
                # Attempt to save to wandb as well (keep local copy)
                try:
                    wandb.save(str(best_path))
                except Exception:
                    logger.exception('wandb.save failed for %s', best_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info("Early stopping triggered after %d epochs", epoch+1)
                    break
        
        # Load best model
        best_path = MODEL_DIR / 'best_stego_model.pth'
        self.model.load_state_dict(torch.load(str(best_path)))
        logger.info("Loaded best model from %s for evaluation", best_path)

    def plot_training_history(self, save_path='training_history.png'):
        """Plot training and validation metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        out = MODEL_DIR / save_path
        plt.savefig(str(out), dpi=300, bbox_inches='tight')
        logger.info("Saved training history to %s", out)

    @weave.op()
    def evaluate(self, test_loader, save_plots=True):
        """Comprehensive evaluation with metrics and visualizations."""
        logger.info("%s", "="*50)
        logger.info("EVALUATION ON TEST SET")
        logger.info("%s", "="*50)
        
        test_loss, test_acc, labels, predictions, probs = self.validate(test_loader)
        
        logger.info("Test Loss: %.4f", test_loss)
        logger.info("Test Accuracy: %.2f%%", test_acc)
        
        # Classification report
        logger.info("%s", "-"*50)
        # Ensure we always report both classes (0=Clean, 1=Steganography).
        # classification_report raises if target_names length doesn't match found classes,
        # so pass explicit labels and a zero_division policy to avoid errors when a class is missing.
        try:
            report = classification_report(
                labels,
                predictions,
                labels=[0, 1],
                target_names=['Clean', 'Steganography'],
                zero_division=0,
            )
        except Exception:
            logger.exception('classification_report failed; falling back to raw label counts')
            report = str({
                'unique_labels': np.unique(labels).tolist(),
                'counts': {int(l): int(labels.count(l)) for l in set(labels)}
            })

        logger.info("Classification Report:\n%s", report)

        # Confusion matrix (force 2x2 layout even if one class missing)
        cm = confusion_matrix(labels, predictions, labels=[0, 1])

        if save_plots:
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Clean', 'Stego'],
                yticklabels=['Clean', 'Stego']
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            out_cm = MODEL_DIR / 'confusion_matrix.png'
            plt.savefig(str(out_cm), dpi=300, bbox_inches='tight')
            logger.info("Saved confusion matrix to %s", out_cm)
            
            # ROC curve - only valid if both classes are present
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                try:
                    fpr, tpr, thresholds = roc_curve(labels, probs)
                    auc_score = roc_auc_score(labels, probs)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    plt.grid(True)
                    out_roc = MODEL_DIR / 'roc_curve.png'
                    plt.savefig(str(out_roc), dpi=300, bbox_inches='tight')
                    logger.info("Saved ROC curve to %s", out_roc)
                    logger.info("AUC Score: %.4f", auc_score)
                except Exception:
                    logger.exception('Failed to compute ROC/AUC')
            else:
                logger.warning('Skipping ROC/AUC: only one class present in labels (%s)', unique_labels)
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'labels': labels,
            'predictions': predictions,
            'probabilities': probs,
            'confusion_matrix': cm
        }


def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Perform stratified split of dataset to maintain class balance across splits.
    
    Args:
        dataset: Dataset with labels accessible via dataset[idx][1]
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as Subset objects
    """
    # Extract all labels
    logger.info("Extracting labels for stratified split...")
    labels = []
    for idx in range(len(dataset)):
        try:
            _, label = dataset[idx]
            labels.append(label)
        except Exception as e:
            logger.warning("Failed to load sample %d for stratification: %s", idx, e)
            labels.append(-1)  # Mark as invalid
    
    labels = np.array(labels)
    indices = np.arange(len(dataset))
    
    # Filter out invalid samples
    valid_mask = labels >= 0
    valid_indices = indices[valid_mask]
    valid_labels = labels[valid_mask]
    
    if len(valid_indices) == 0:
        raise RuntimeError("No valid samples found in dataset")
    
    # Count class distribution
    unique, counts = np.unique(valid_labels, return_counts=True)
    logger.info("Overall class distribution:")
    for cls, count in zip(unique, counts):
        logger.info("  Class %d: %d samples (%.1f%%)", cls, count, 100 * count / len(valid_labels))
    
    # First split: train vs (val+test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        valid_indices,
        valid_labels,
        train_size=train_ratio,
        stratify=valid_labels,
        random_state=seed
    )
    
    # Second split: val vs test (from temp)
    # Calculate proportion: if we have 0.15 val and 0.15 test from total,
    # and temp is 0.30 of total, then val should be 0.15/0.30 = 0.5 of temp
    val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices,
        temp_labels,
        train_size=val_ratio_of_temp,
        stratify=temp_labels,
        random_state=seed
    )
    
    # Log split statistics
    logger.info("Stratified split complete:")
    logger.info("  Train: %d samples", len(train_indices))
    for cls in unique:
        count = np.sum(train_labels == cls)
        logger.info("    Class %d: %d (%.1f%%)", cls, count, 100 * count / len(train_labels))
    
    logger.info("  Val: %d samples", len(val_indices))
    for cls in unique:
        count = np.sum(val_labels == cls)
        logger.info("    Class %d: %d (%.1f%%)", cls, count, 100 * count / len(val_labels))
    
    logger.info("  Test: %d samples", len(test_indices))
    for cls in unique:
        count = np.sum(test_labels == cls)
        logger.info("    Class %d: %d (%.1f%%)", cls, count, 100 * count / len(test_labels))
    
    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training pipeline."""
    
    # Configuration
    EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'
    IMG_ROOT = Path('.')
    BATCH_SIZE = 16
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 20
    LEARNING_RATE = 53-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # If using CUDA, enable cuDNN autotuner for potentially faster kernels
    if DEVICE.startswith('cuda'):
        try:
            torch.backends.cudnn.benchmark = True
            logger.info('Enabled torch.backends.cudnn.benchmark')
        except Exception:
            pass
    
    logger.info("Using device: %s", DEVICE)

    # Import the dataset
    dataset = StegoImageDataset(
        EXCEL_PATH, 
        IMG_ROOT, 
        dct_channels=['Y', 'Cb', 'Cr']
    )
    
    # Get feature dimension from first sample
    # Find first valid sample to get input_dim
    # Helps avoid errors if there is a corrupted image, i.e. 0 features
    input_dim = None
    for idx in range(min(100, len(dataset))):  # Check first 100 samples
        try:
            features, _ = dataset[idx]
            if features.numel() > 0:  # Check not empty
                input_dim = features.shape[0]
                logger.info("Feature dimension: %d (from sample %d)", input_dim, idx)
                break
        except Exception as e:
            logger.warning("Failed to load sample %d: %s", idx, e)
            continue
    
    if input_dim is None or input_dim == 0:
        raise RuntimeError("Could not determine feature dimension - all samples failed!")
    logger.info("Feature dimension: %d", input_dim)

    # Initialize wandb and push hyperparameters
    wandb_run = None
    if wandb is not None:
        try:
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'stegnography-detector'),
                name=os.environ.get('WANDB_RUN_NAME', 'obvious_and_subtle_detection'),
                config={
                    'batch_size': BATCH_SIZE,
                    'epochs': EPOCHS,
                    'learning_rate': LEARNING_RATE,
                    'device': DEVICE,
                }
            )
            wandb_run = wandb.run
            # push derived values
            try:
                wandb.config.update({'input_dim': input_dim, 'dataset_size': len(dataset)}, allow_val_change=True)
            except Exception:
                logger.exception('Failed to update wandb.config')
        except Exception:
            logger.exception('wandb.init failed; continuing without wandb')
    
    # Stratified split to maintain class balance
    train_dataset, val_dataset, test_dataset = stratified_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Create data loaders. Use environment variable NUM_WORKERS to tune parallelism
    num_workers = int(os.environ.get('NUM_WORKERS', '0'))
    pin_memory = True if DEVICE.startswith('cuda') else False

    # Log chosen loader settings
    logger.info('DataLoader settings: num_workers=%d, pin_memory=%s, batch_size=%d', num_workers, pin_memory, BATCH_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    logger.info("Dataset splits: Train=%d Val=%d Test=%d", len(train_dataset), len(val_dataset), len(test_dataset))
    
    # Initialize model
    model = StegoDetectionCNN(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.5
    )

    # Initialize wandb (optional)
    wandb.init(
        entity="Anti_Forensic_Stegnography_Detection",
        project="stegnography-detection",
        name=os.environ.get('WANDB_RUN_NAME', 'obvious_and_subtle_detection'),
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "CNN",
            "dataset": "steg_excel",
            "epochs": EPOCHS,
        }
    )
    wandb_run = wandb.run
    logger.info('Initialized wandb run: %s', getattr(wandb_run, 'name', None))
    
    logger.info("Model architecture:\n%s", model)
    logger.info("Total parameters: %d", sum(p.numel() for p in model.parameters()))
    # If wandb is active, watch model
    if wandb is not None and wandb_run is not None:
        try:
            wandb.watch(model, log='all', log_freq=100)
        except Exception:
            logger.exception('wandb.watch failed')
    
    # Initialize trainer
    trainer = StegoDetectionTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    # =========== MODEL TRAINING ============ #
    # Train the model
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    results = trainer.evaluate(test_loader)
    
    # Save final results
    final_path = MODEL_DIR / 'final_stego_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'results': results
    }, str(final_path))
    logger.info("Saved final model to %s", final_path)
    # Also attempt to save final model to wandb
    if wandb is not None and wandb_run is not None:
        try:
            wandb.save(str(final_path))
        except Exception:
            logger.exception('wandb.save failed for %s', final_path)


def predict_single_image(model_path, image_path, dataset_config):
    """
    Predict whether a single image contains steganography.
    
    Args:
        model_path: Path to saved model
        image_path: Path to image to analyze
        dataset_config: Dictionary with dataset configuration
    """
    # Load model
    checkpoint = torch.load(model_path)
    input_dim = checkpoint['input_dim']
    
    model = StegoDetectionCNN(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Load and process image (you'll need to adapt this to your dataset class)
    # from your_module import StegoImageDataset
    # dataset = StegoImageDataset(**dataset_config)
    # features = dataset._extract_dct_features(Path(image_path), None)
    
    # Make prediction
    # features = features.unsqueeze(0).to(device)  # Add batch dimension
    # with torch.no_grad():
    #     outputs = model(features)
    #     probs = torch.softmax(outputs, dim=1)
    #     predicted_class = torch.argmax(probs, dim=1).item()
    #     confidence = probs[0, predicted_class].item()
    
    # result = {
    #     'prediction': 'Steganography Detected' if predicted_class == 1 else 'Clean Image',
    #     'confidence': confidence * 100,
    #     'stego_probability': probs[0, 1].item() * 100,
    #     'clean_probability': probs[0, 0].item() * 100
    # }
    
    # return result
    pass


if __name__ == "__main__":
    main()
