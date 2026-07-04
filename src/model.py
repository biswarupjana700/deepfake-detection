import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeDetector(nn.Module):
    def __init__(self, model_name='mobilenet_v2', num_classes=2, pretrained=True):
        """
        Initialize the DeepFake Detection Model
        
        Args:
            model_name: Name of pretrained model ('mobilenet_v2' or 'resnet18')
            num_classes: Number of output classes (2: real/fake)
            pretrained: Whether to use pretrained weights
        """
        super(DeepFakeDetector, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'mobilenet_v2':
            # Load MobileNetV2
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
            # Replace classifier
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'resnet18':
            # Load ResNet18
            self.base_model = models.resnet18(pretrained=pretrained)
            # Replace classifier
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Model {model_name} not supported. Use 'mobilenet_v2' or 'resnet18'")
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Output logits
        """
        x = self.base_model(x)
        x = self.dropout(x)
        return x

def freeze_layers(model, freeze_until='classifier'):
    """
    Freeze layers of the model
    
    Args:
        model: Model instance
        freeze_until: Freeze layers until this point
    """
    for name, param in model.named_parameters():
        if freeze_until in name:
            break
        param.requires_grad = False

def get_model(model_name='mobilenet_v2', num_classes=2, freeze=True):
    """
    Get the model ready for training
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        freeze: Whether to freeze pretrained layers
    
    Returns:
        Model instance
    """
    model = DeepFakeDetector(model_name, num_classes, pretrained=True)
    
    if freeze:
        if model_name == 'mobilenet_v2':
            freeze_layers(model, 'classifier')
        elif model_name == 'resnet18':
            freeze_layers(model, 'fc')
    
    return model

def count_parameters(model):
    """Count trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return total_params, trainable_params

if __name__ == "__main__":
    print("="*60)
    print("🤖 MODEL ARCHITECTURE")
    print("="*60)
    
    # Test MobileNetV2
    print("\n📱 MobileNetV2:")
    model1 = get_model('mobilenet_v2', num_classes=2, freeze=True)
    count_parameters(model1)
    
    # Test ResNet18
    print("\n🔷 ResNet18:")
    model2 = get_model('resnet18', num_classes=2, freeze=True)
    count_parameters(model2)
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model1(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    print("\n✅ Model test complete!")
    print("="*60)