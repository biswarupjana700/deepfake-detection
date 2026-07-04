import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_test_dataset():
    """Create a small test dataset with fake and real images"""
    
    print("🎨 Creating test dataset...")
    print("="*60)
    
    # Base directories
    base_dir = "D:/DeepFake-Detection/dataset"
    splits = ['train', 'val', 'test']
    categories = ['real', 'fake']
    
    # Create directories
    for split in splits:
        for category in categories:
            os.makedirs(f"{base_dir}/{split}/{category}", exist_ok=True)
    
    print("✅ Created folder structure")
    
    # Generate sample images
    print("🖼️ Generating sample images...")
    
    # Different number of images per split
    images_per_split = {
        'train': 80,
        'val': 30,
        'test': 40
    }
    
    for split in splits:
        num_images = images_per_split[split]
        
        for category in categories:
            for i in range(num_images):
                # Create random image
                img = Image.new('RGB', (224, 224), color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ))
                
                # Add some features to make it look like a face
                draw = ImageDraw.Draw(img)
                
                # Draw a face-like shape
                if category == 'real':
                    # Real images: more natural colors
                    face_color = (random.randint(180, 230), 
                                 random.randint(130, 190), 
                                 random.randint(100, 160))
                else:
                    # Fake images: more artificial colors
                    face_color = (random.randint(0, 100), 
                                 random.randint(0, 100), 
                                 random.randint(0, 100))
                
                # Draw face circle
                center_x = random.randint(80, 144)
                center_y = random.randint(80, 144)
                radius = random.randint(40, 70)
                draw.ellipse([center_x-radius, center_y-radius, 
                             center_x+radius, center_y+radius], 
                           fill=face_color)
                
                # Draw eyes
                eye_color = (0, 0, 0)
                draw.ellipse([center_x-30, center_y-20, 
                             center_x-15, center_y-5], fill=eye_color)
                draw.ellipse([center_x+15, center_y-20, 
                             center_x+30, center_y-5], fill=eye_color)
                
                # Draw mouth
                draw.arc([center_x-20, center_y+5, 
                         center_x+20, center_y+25], 
                        start=0, end=180, fill=(150, 50, 50), width=3)
                
                # Add text label
                draw.text((10, 10), category.upper(), 
                         fill=(255, 255, 255))
                
                # Save image
                filename = f"{category}_{split}_{i:04d}.jpg"
                filepath = f"{base_dir}/{split}/{category}/{filename}"
                img.save(filepath)
    
    print(f"✅ Generated sample images")
    print(f"📁 Dataset ready at: {base_dir}")
    
    # Show summary
    print("\n📊 Dataset Summary:")
    total = 0
    for split in splits:
        for category in categories:
            count = len(os.listdir(f"{base_dir}/{split}/{category}"))
            print(f"   {split}/{category}: {count} images")
            total += count
    print(f"\n📈 Total images: {total}")

if __name__ == "__main__":
    create_test_dataset()