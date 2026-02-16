"""
ThermoVision - Fire Detection Model Trainer
Train CNN on fire/no-fire dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import sys
import time

# Import our CNN model
sys.path.append(str(Path(__file__).parent.parent))
from detection.fire_detector import FireCNN


class FireDataset(Dataset):
    """
    PyTorch dataset for fire detection
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root directory with 'fire' and 'no_fire' subdirectories
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Collect all image paths
        self.samples = []

        # Fire images (label = 1)
        fire_dir = self.data_dir / "fire"
        if fire_dir.exists():
            for img_path in fire_dir.glob("*.jpg"):
                self.samples.append((img_path, 1))

        # No-fire images (label = 0)
        no_fire_dir = self.data_dir / "no_fire"
        if no_fire_dir.exists():
            for img_path in no_fire_dir.glob("*.jpg"):
                self.samples.append((img_path, 0))

        print(f"📊 Dataset loaded: {len(self.samples)} images")
        fire_count = sum(1 for _, label in self.samples if label == 1)
        print(f"   Fire: {fire_count}, No-fire: {len(self.samples) - fire_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class FireModelTrainer:
    """
    Train fire detection CNN
    """

    def __init__(self, data_dir, model_save_path="models/fire_detector.pth"):
        """
        Args:
            data_dir: Directory containing fire/no_fire subdirectories
            model_save_path: Where to save trained model
        """
        self.data_dir = Path(data_dir)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Training device: {self.device}")

        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Model
        self.model = None
        self.optimizer = None
        self.criterion = None

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def prepare_data(self, train_split=0.8, batch_size=32):
        """
        Prepare data loaders

        Args:
            train_split: Fraction of data for training
            batch_size: Batch size
        """
        # Load full dataset
        full_dataset = FireDataset(self.data_dir, transform=self.train_transform)

        # Split into train/val
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Update validation transform
        val_dataset.dataset.transform = self.val_transform

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        print(f"✅ Data prepared: {train_size} train, {val_size} val")

    def build_model(self, learning_rate=0.001):
        """
        Build and compile model

        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model = FireCNN().to(self.device)

        # Loss function (weighted for class imbalance if needed)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        print("✅ Model built")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate model"""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc

    def train(self, epochs=20):
        """
        Full training loop

        Args:
            epochs: Number of epochs to train
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        best_val_acc = 0.0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")

            print()

        print("=" * 60)
        print(f"Training Complete! Best Val Acc: {best_val_acc:.2f}%")
        print("=" * 60)

    def save_model(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

        torch.save(checkpoint, self.model_save_path)
        print(f"💾 Model saved to {self.model_save_path}")

    def load_model(self):
        """Load saved model"""
        if not self.model_save_path.exists():
            print(f"❌ No model found at {self.model_save_path}")
            return False

        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✅ Model loaded from {self.model_save_path}")
        return True


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(description="Train Fire Detection Model")
    parser.add_argument('--data_dir', type=str, default='data/fire_dataset',
                        help='Directory with fire/no_fire subdirectories')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("💡 Run collect_fire_data.py first to collect training data")
        return

    # Initialize trainer
    trainer = FireModelTrainer(args.data_dir)

    # Prepare data
    trainer.prepare_data(batch_size=args.batch_size)

    # Build model
    trainer.build_model(learning_rate=args.lr)

    # Train
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()