import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import random

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), augment=True):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (width, height)
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.augment = augment
        
        # Define transformations for training (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Define transformations for validation/test (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def process_image(self, image_path, is_train=True):
        """Process a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            if is_train:
                return self.train_transform(image)
            else:
                return self.val_transform(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_dataset(self, data_path, output_path, max_images=None):
        """
        Process entire dataset
        
        Args:
            data_path: Path to dataset (with train/val/test folders)
            output_path: Path to save processed data
            max_images: Maximum images to process per split (None = all)
        """
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        splits = ['train', 'val', 'test']
        results = {}
        
        for split in splits:
            print(f"\n📁 Processing {split} set...")
            split_path = os.path.join(data_path, split)
            
            if not os.path.exists(split_path):
                print(f"⚠️ Warning: {split_path} not found")
                continue
            
            all_tensors = []
            all_labels = []
            
            # Process each category
            for category in ['real', 'fake']:
                category_path = os.path.join(split_path, category)
                if not os.path.exists(category_path):
                    print(f"⚠️ Warning: {category_path} not found")
                    continue
                
                # Get all images
                image_files = [f for f in os.listdir(category_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if max_images:
                    image_files = image_files[:max_images]
                
                # Determine if training (apply augmentation)
                is_train = (split == 'train')
                
                print(f"   Processing {len(image_files)} {category} images...")
                
                for image_file in tqdm(image_files, desc=f"     {category}"):
                    image_path = os.path.join(category_path, image_file)
                    
                    # Process image
                    tensor = self.process_image(image_path, is_train)
                    
                    if tensor is not None:
                        all_tensors.append(tensor)
                        label = 0 if category == 'real' else 1  # 0: real, 1: fake
                        all_labels.append(label)
            
            if all_tensors:
                # Convert to tensors
                tensors = torch.stack(all_tensors)
                labels = torch.tensor(all_labels)
                
                # Save
                save_path = os.path.join(output_path, f'{split}_data.pt')
                torch.save({
                    'tensors': tensors,
                    'labels': labels
                }, save_path)
                
                print(f"   ✅ Saved {len(tensors)} images to {save_path}")
                results[split] = {'tensors': tensors, 'labels': labels}
            else:
                print(f"   ❌ No images processed for {split}")
        
        return results
    
    def create_dataloader(self, data, batch_size=32, shuffle=True):
        """
        Create PyTorch DataLoader from processed data
        """
        if not data:
            return None
        
        tensors = data['tensors']
        labels = data['labels']
        
        dataset = torch.utils.data.TensorDataset(tensors, labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Windows compatibility
        )
        
        return loader

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for deepfake detection')
    parser.add_argument('--dataset', type=str, default='./dataset_small',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='./processed_data',
                       help='Path to output directory')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (width height)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum images to process per split (None = all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for dataloader')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🖼️ IMAGE PREPROCESSING")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Image size: {args.size[0]}x{args.size[1]}")
    print(f"Max images per split: {args.max_images or 'All'}")
    print("="*60)
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=tuple(args.size),
        augment=True
    )
    
    # Process dataset
    processed_data = preprocessor.process_dataset(
        data_path=args.dataset,
        output_path=args.output,
        max_images=args.max_images
    )
    
    # Create dataloaders
    print("\n📦 Creating dataloaders...")
    train_loader = preprocessor.create_dataloader(
        processed_data.get('train'), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = preprocessor.create_dataloader(
        processed_data.get('val'), 
        batch_size=args.batch_size, 
        shuffle=False
    )
    test_loader = preprocessor.create_dataloader(
        processed_data.get('test'), 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Summary
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    if processed_data.get('train'):
        print(f"Train: {len(processed_data['train']['tensors'])} images")
    if processed_data.get('val'):
        print(f"Validation: {len(processed_data['val']['tensors'])} images")
    if processed_data.get('test'):
        print(f"Test: {len(processed_data['test']['tensors'])} images")
    print(f"\n📁 Processed data saved to: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()