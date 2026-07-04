import os
import kagglehub
import shutil
from pathlib import Path

def download_deepfake_dataset():
    """Download a sample deepfake dataset"""
    
    print("📥 Downloading DeepFake Dataset...")
    print("="*60)
    
    # Try to download a small dataset
    try:
        # This downloads a sample dataset
        path = kagglehub.dataset_download("sanjaykumarsingh/deepfake-video-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Copy to your project
        destination = "D:/DeepFake-Detection/dataset"
        
        # Move files to proper structure
        if os.path.exists(path):
            print("📁 Organizing dataset...")
            # You'll need to organize based on the downloaded structure
            
        return path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n📝 Alternative: Download manually from Kaggle")
        print("1. Go to: https://www.kaggle.com/datasets/sanjaykumarsingh/deepfake-video-dataset")
        print("2. Click Download")
        print("3. Extract to D:/DeepFake-Detection/dataset")
        return None

if __name__ == "__main__":
    download_deepfake_dataset()