# Author: Kaleb Austgen
# Date: 10/20/25
# Purpose: Fully interconnected network to train 1D vectors based off of DCT coefficiencts from JPEGS to discover stenography

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
from sten_dct import StegoImageDataset
import logging
import os
# Directory where this script lives; used for saving/loading models and plots
MODEL_DIR = Path(__file__).parent

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

class StegoDetectionTrainer:
    """Handles training, validation, and evaluation of the stego detection model."""

    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs= []

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
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

    def train(self, train_loader, val_loader, epochs=50, early_stop_patience=10):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch+1, epochs)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            logger.info("Train Loss: %.4f, Train Acc: %.2f%%", train_loss, train_acc)
            logger.info("Val Loss: %.4f, Val Acc: %.2f%%", val_loss, val_acc)
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
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
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
        logger.info("Classification Report:\n%s", classification_report(
            labels, 
            predictions, 
            target_names=['Clean', 'Steganography']
        ))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
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
            
            # ROC curve
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
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'labels': labels,
            'predictions': predictions,
            'probabilities': probs,
            'confusion_matrix': cm
        }

def main():
    """Main training pipeline."""
    
    # Configuration
    EXCEL_PATH = Path(__file__).parent.parent / 'dataGen' / 'stego_training.xlsx'
    IMG_ROOT = Path('.')
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
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
        dct_channels=['Y', 'Db', 'Cr']
    )
    
    # Get feature dimension from first sample
    features, _ = dataset[0]
    input_dim = features.shape[0]
    logger.info("Feature dimension: %d", input_dim)

    # Initialize wandb (optional) and push hyperparameters
    wandb_run = None
    if wandb is not None:
        try:
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'timestomping-detector'),
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
    
    # Split dataset: 70% train, 15% val, 15% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the datasets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
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
    
    # Train the model
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stop_patience=15
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
