import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import sys

# Add src to path so we can import model
sys.path.append('src')
from model import get_model

def train_improved_model():
    """Train model with improvements for better accuracy"""
    
    print("="*60)
    print("🚀 IMPROVED MODEL TRAINING")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n📂 Loading data...")
    
    # Check if processed data exists
    if not os.path.exists('processed_data/train_data.pt'):
        print("❌ Error: Processed data not found!")
        print("Please run preprocessing first:")
        print("python src/preprocess.py --dataset ./dataset_small --output ./processed_data")
        return
    
    train_data = torch.load('processed_data/train_data.pt')
    val_data = torch.load('processed_data/val_data.pt')
    test_data = torch.load('processed_data/test_data.pt')
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_data['tensors'], train_data['labels'])
    val_dataset = torch.utils.data.TensorDataset(val_data['tensors'], val_data['labels'])
    test_dataset = torch.utils.data.TensorDataset(test_data['tensors'], test_data['labels'])
    
    # Larger batch size for better gradients
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    # Use ResNet18 (better for face detection)
    print("\n🤖 Loading ResNet18 model...")
    model = get_model('resnet18', num_classes=2, freeze=False)
    model = model.to(device)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=25)
    
    # Label smoothing helps prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training settings
    epochs = 25
    best_acc = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("-"*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs}")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'models/best_model_improved.pth')
            print(f"   ✅ New best model saved! Acc: {best_acc:.2f}%")
        
        print("-"*60)
    
    print(f"\n🎉 Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
    
    # Test the model
    print("\n🧪 Testing best model...")
    checkpoint = torch.load('models/best_model_improved.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    test_f1 = f1_score(all_labels, all_preds)
    
    print(f"📊 Test Results:")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   F1 Score: {test_f1:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_improved.png')
    plt.show()
    
    print(f"\n✅ Improved training complete!")
    print(f"📁 Model saved to: models/best_model_improved.pth")
    print(f"📊 Plot saved to: results/training_improved.png")
    
    return model, history

if __name__ == "__main__":
    train_improved_model()