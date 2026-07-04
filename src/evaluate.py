import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append('src')
from model import get_model

def evaluate_model():
    """Evaluate the trained model on test data"""
    
    print("="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("\n📂 Loading test data...")
    
    # Check if processed data exists
    if not os.path.exists('processed_data/test_data.pt'):
        print("❌ Error: Processed data not found!")
        print("Please run preprocessing first:")
        print("python src/preprocess.py --dataset ./dataset_small --output ./processed_data")
        return
    
    test_data = torch.load('processed_data/test_data.pt')
    
    # Create dataloader
    test_dataset = torch.utils.data.TensorDataset(test_data['tensors'], test_data['labels'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print("\n🤖 Loading trained model...")
    model = get_model('resnet18', num_classes=2, freeze=False)
    
    # Load the best model
    if not os.path.exists('models/best_model_improved.pth'):
        print("❌ Error: Model file not found!")
        print("Please train the model first:")
        print("python src/train_improved.py")
        return
    
    checkpoint = torch.load('models/best_model_improved.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Get predictions
    print("\n🧪 Evaluating model on test set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            _, preds = torch.max(output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability of FAKE (class 1)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {auc:.4f}")
    
    print("\n📊 Confusion Matrix:")
    print(f"            Predicted")
    print(f"            Real  Fake")
    print(f"   Real     [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"   Fake     [{cm[1][0]:4d}  {cm[1][1]:4d}]")
    
    # Create visualizations
    os.makedirs('results', exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    print("\n📁 Confusion matrix saved to: results/confusion_matrix.png")
    plt.show()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curve.png')
    print("📁 ROC curve saved to: results/roc_curve.png")
    plt.show()
    
    # 3. Save results to file
    with open('results/evaluation_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"ROC AUC:   {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"[[{cm[0][0]}, {cm[0][1]}]\n")
        f.write(f" [{cm[1][0]}, {cm[1][1]}]]\n")
    
    print("📁 Results saved to: results/evaluation_results.txt")
    
    # Print summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   ROC AUC: {auc:.4f}")
    print(f"   Confusion Matrix: {cm[0][0]} real correct, {cm[1][1]} fake correct")

if __name__ == "__main__":
    evaluate_model()