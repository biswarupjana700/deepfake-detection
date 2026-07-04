import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Create small dataset with 2000 images per category per split
for split in ['train', 'val', 'test']:
    for category in ['real', 'fake']:
        src = f'dataset/{split}/{category}'
        dst = f'dataset_small/{split}/{category}'
        
        # Create destination folder
        os.makedirs(dst, exist_ok=True)
        
        # Get all images
        images = os.listdir(src)
        
        # Take first 2000 (or all if less)
        num_images = min(2000, len(images))
        selected = sorted(images)[:num_images]
        
        # Copy images
        for img in selected:
            shutil.copy2(f'{src}/{img}', f'{dst}/{img}')
        
        print(f'{split}/{category}: {len(selected)} images')

print("\n✅ Dataset subset created successfully!")
print("📊 Total images: 12,000 (6,000 real + 6,000 fake)")