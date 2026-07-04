import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_f1

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_f1

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cuda'):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Starting training on {device}")
    print(f"Epochs: {epochs}, Learning rate: {lr}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✓ New best model saved! Validation accuracy: {val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation accuracy: {best_val_acc:.4f}")
    
    return model, history

def plot_training_history(history, save_path='./results/training_history.png'):
    """Plot training history"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history['train_acc'], label='Train Accuracy')
    axes[0].plot(history['val_acc'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history plot saved to {save_path}")

def save_model(model, path='./models/best_model.pth'):
    """Save model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'num_classes': 2
    }, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from model import get_model
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load processed data
    processed_data_path = './processed_data'
    if not os.path.exists(processed_data_path):
        print("Error: Processed data not found. Run preprocess.py first.")
        exit()
    
    # Load processed data
    train_data = torch.load(os.path.join(processed_data_path, 'train_data.pt'))
    val_data = torch.load(os.path.join(processed_data_path, 'val_data.pt'))
    test_data = torch.load(os.path.join(processed_data_path, 'test_data.pt'))
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_data['tensors'], train_data['labels'])
    val_dataset = torch.utils.data.TensorDataset(val_data['tensors'], val_data['labels'])
    test_dataset = torch.utils.data.TensorDataset(test_data['tensors'], test_data['labels'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get model (choose MobileNetV2 for faster training)
    model = get_model('mobilenet_v2', num_classes=2, freeze=True)
    print(model)
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=10, lr=0.001, device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(model)
    
    print("\n✅ Training complete!")