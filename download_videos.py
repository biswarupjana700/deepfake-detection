import os
import requests
from tqdm import tqdm
import zipfile
import io

def download_video(url, filename):
    """Download a video from URL"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(filename, 'wb') as f:
                for data in tqdm(response.iter_content(chunk_size=1024), 
                               total=total_size//1024, 
                               unit='KB',
                               desc=f"Downloading {os.path.basename(filename)}"):
                    f.write(data)
            return True
        else:
            print(f"❌ Failed to download: {url}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def download_sample_videos():
    """Download sample videos for testing"""
    
    print("📥 Downloading sample videos...")
    print("="*60)
    
    # Create directories
    base_dir = "D:/DeepFake-Detection/dataset"
    splits = ['train', 'val', 'test']
    categories = ['real', 'fake']
    
    for split in splits:
        for category in categories:
            os.makedirs(f"{base_dir}/{split}/{category}", exist_ok=True)
    
    # Sample video URLs (working, small videos for testing)
    # These are from sample-videos.com - safe, small, and reliable
    sample_videos = [
        {
            "url": "https://sample-videos.com/video321/mp4/240/big_buck_bunny_240p_1mb.mp4",
            "name": "bunny_1mb.mp4",
            "category": "real",
            "split": "train"
        },
        {
            "url": "https://sample-videos.com/video321/mp4/240/big_buck_bunny_240p_2mb.mp4",
            "name": "bunny_2mb.mp4",
            "category": "real",
            "split": "train"
        },
        {
            "url": "https://sample-videos.com/video321/mp4/240/big_buck_bunny_240p_3mb.mp4",
            "name": "bunny_3mb.mp4",
            "category": "fake",
            "split": "train"
        },
        {
            "url": "https://sample-videos.com/video321/mp4/240/big_buck_bunny_240p_5mb.mp4",
            "name": "bunny_5mb.mp4",
            "category": "fake",
            "split": "val"
        }
    ]
    
    print("\n📹 Downloading sample videos...")
    print("(Note: These are sample videos, not actual deepfakes)")
    print("They will help you test the frame extraction pipeline)\n")
    
    for video in sample_videos:
        filename = f"{base_dir}/{video['split']}/{video['category']}/{video['name']}"
        print(f"\n📥 Downloading: {video['name']}")
        success = download_video(video['url'], filename)
        if success:
            print(f"✅ Downloaded: {filename}")
        else:
            print(f"⚠️ Skipping: {video['name']}")
    
    print("\n" + "="*60)
    print("✅ Video download complete!")
    print("\n📁 Video locations:")
    print("   Real videos: dataset/train/real/")
    print("   Fake videos: dataset/train/fake/")
    print("   (Files labeled as 'fake' are just for testing the pipeline)")
    
    print("\n💡 Tip: For real deepfake detection, you would need:")
    print("   1. Real face videos (actors talking)")
    print("   2. Deepfake videos (same actors manipulated)")
    print("   Download from Kaggle: https://www.kaggle.com/datasets")

if __name__ == "__main__":
    download_sample_videos()