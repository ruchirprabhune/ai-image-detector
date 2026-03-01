"""
Deep Learning Model Training Script
Trains EfficientNet model to classify real vs AI-generated images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json

class ImageDataset(Dataset):
    """Dataset for real vs AI images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class AIImageClassifier(nn.Module):
    """EfficientNet-based classifier for AI image detection"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(AIImageClassifier, self).__init__()
        
        # Load pre-trained EfficientNet-B0 (smallest, fastest)
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b0')
        
        # Get the number of features from the last layer
        num_features = self.backbone._fc.in_features
        
        # Replace classifier
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.class_names = ['Real', 'Stable_Diffusion', 'Midjourney', 'DALLE', 'Unknown']
        
    def train(self, train_loader, val_loader, epochs=10, learning_rate=0.001):
        """Train the model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                print("✓ Saved best model")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
        return history
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/(pbar.n+1), 
                            'acc': 100.*correct/total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def predict(self, image_path):
        """Predict class for a single image"""
        self.model.eval()
        
        transform = get_transforms(train=False)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        return {
            'class': self.class_names[predicted.item()],
            'confidence': confidence.item() * 100,
            'probabilities': {
                self.class_names[i]: probabilities[0][i].item() * 100 
                for i in range(len(self.class_names))
            }
        }
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint['class_names']
        self.model.eval()

def get_transforms(train=True):
    """Get image transforms"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def prepare_dataset(data_dir):
    """
    Prepare dataset from directory structure:
    data_dir/
        real/
        stable_diffusion/
        midjourney/
        dalle/
        unknown/
    """
    image_paths = []
    labels = []
    
    class_names = ['real', 'stable_diffusion', 'midjourney', 'dalle', 'unknown']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    return image_paths, labels

def main():
    """Main training function"""
    
    # Configuration
    DATA_DIR = '../datasets/train'
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    image_paths, labels = prepare_dataset(DATA_DIR)
    
    if len(image_paths) == 0:
        print("Error: No images found in dataset directory!")
        print(f"Please organize your data in: {DATA_DIR}")
        print("Structure: data_dir/real/, data_dir/stable_diffusion/, etc.")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Split into train and validation
    split_idx = int(0.8 * len(image_paths))
    train_paths = image_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_paths = image_paths[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create datasets and loaders
    train_dataset = ImageDataset(train_paths, train_labels, 
                                 transform=get_transforms(train=True))
    val_dataset = ImageDataset(val_paths, val_labels, 
                               transform=get_transforms(train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    # Create model
    print("\nInitializing model...")
    model = AIImageClassifier(num_classes=5, pretrained=True)
    trainer = ModelTrainer(model, device=DEVICE)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader, 
                           epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    # Save final model
    trainer.save_model('../models/final_model.pth')
    
    # Save training history
    with open('../models/training_history.json', 'w') as f:
        json.dump(history, f)
    
    print("\n✓ Training complete!")
    print(f"Best model saved to: ../models/best_model.pth")
    print(f"Final model saved to: ../models/final_model.pth")

if __name__ == '__main__':
    main()
