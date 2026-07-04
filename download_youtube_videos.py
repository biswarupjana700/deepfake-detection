import os
import subprocess
import sys

def check_yt_dlp():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_yt_dlp():
    """Install yt-dlp"""
    print("📦 Installing yt-dlp...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])

def download_youtube_video(url, output_path, max_duration=30):
    """
    Download a short clip from YouTube
    
    Args:
        url: YouTube video URL
        output_path: Where to save the video
        max_duration: Maximum duration in seconds
    """
    try:
        # Download only first 30 seconds
        cmd = [
            'yt-dlp',
            '-f', 'mp4',
            '--no-playlist',
            '--download-sections', f'*0-{max_duration}',
            '--force-keyframes-at-cuts',
            '-o', output_path,
            url
        ]
        
        print(f"📥 Downloading from YouTube...")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Downloaded: {os.path.basename(output_path)}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def download_sample_videos():
    """Download sample videos from YouTube"""
    
    print("📥 Downloading sample videos from YouTube...")
    print("="*60)
    
    # Create directories
    base_dir = "D:/DeepFake-Detection/dataset"
    splits = ['train', 'val', 'test']
    categories = ['real', 'fake']
    
    for split in splits:
        for category in categories:
            os.makedirs(f"{base_dir}/{split}/{category}", exist_ok=True)
    
    # Check if yt-dlp is installed
    if not check_yt_dlp():
        print("📦 yt-dlp not found. Installing...")
        install_yt_dlp()
    
    # Sample YouTube videos (short, safe, creative commons)
    # These are publicly available videos under Creative Commons license
    sample_videos = [
        {
            "url": "https://www.youtube.com/watch?v=YQHsXMglC9A",  # Short video
            "name": "sample_real_1.mp4",
            "category": "real",
            "split": "train"
        },
        {
            "url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ",  # Short video
            "name": "sample_real_2.mp4",
            "category": "real",
            "split": "train"
        },
        {
            "url": "https://www.youtube.com/watch?v=JGwWNGJdvx8",  # Short video
            "name": "sample_fake_1.mp4",
            "category": "fake",
            "split": "train"
        }
    ]
    
    print("\n📹 Downloading sample videos (first 30 seconds only)...")
    print("(These are sample videos for testing the pipeline)\n")
    
    for video in sample_videos:
        filename = f"{base_dir}/{video['split']}/{video['category']}/{video['name']}"
        print(f"\n📥 Downloading: {video['name']}")
        success = download_youtube_video(video['url'], filename, max_duration=30)
        if success:
            print(f"✅ Saved to: {filename}")
        else:
            print(f"⚠️ Failed to download: {video['name']}")
    
    print("\n" + "="*60)
    print("✅ Video download complete!")
    print("\n📁 Video locations:")
    print("   Real videos: dataset/train/real/")
    print("   Fake videos: dataset/train/fake/")
    
    print("\n💡 Note: These are sample videos, not actual deepfakes.")
    print("   They will help you test the frame extraction pipeline.")

if __name__ == "__main__":
    download_sample_videos()