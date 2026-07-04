import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.append('src')
from model import get_model

class VideoPredictor:
    def __init__(self, model_path='models/best_model_improved.pth'):
        """Initialize the video predictor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = get_model('resnet18', num_classes=2, freeze=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Face detection - try different cascade paths
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        ]
        
        self.face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    print(f"✅ Face cascade loaded from: {path}")
                    break
        
        if self.face_cascade is None or self.face_cascade.empty():
            print("⚠️ Warning: Face cascade could not be loaded. Face detection may not work.")
            # Create a dummy cascade classifier
            self.face_cascade = cv2.CascadeClassifier()
    
    def extract_frames(self, video_path, frame_interval=10, max_frames=50):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            return []
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Video: {os.path.basename(video_path)}")
        print(f"   Total frames: {total_frames}, FPS: {fps}")
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Extracted {len(frames)} frames")
        return frames
    
    def detect_faces(self, frame):
        """Detect and crop faces from frame"""
        if self.face_cascade is None or self.face_cascade.empty():
            # If cascade is not loaded, return the whole frame
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Take the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        padding = int(0.2 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Crop face
        face = frame[y:y+h, x:x+w]
        return face
    
    def predict_frame(self, face):
        """Predict if a face is real or fake"""
        # Convert to PIL and preprocess
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        tensor = self.transform(pil_image).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            _, pred = torch.max(output, 1)
        
        confidence = probs[0][pred.item()].item()
        label = "REAL" if pred.item() == 0 else "FAKE"
        
        return label, confidence
    
    def predict_image(self, image_path):
        """Predict if a single image is real or fake"""
        print(f"\n🖼️ Processing image: {image_path}")
        print("="*60)
        
        # Check if file exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Could not open image: {e}"}
        
        # Predict
        # Convert PIL to numpy for face detection
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        face = self.detect_faces(img_np)
        if face is None:
            # If no face detected, use the whole image
            face = img_np
        
        label, confidence = self.predict_frame(face)
        
        results = {
            "image_path": image_path,
            "prediction": label,
            "confidence": confidence
        }
        
        print(f"🎯 Prediction: {label}")
        print(f"📊 Confidence: {confidence:.2%}")
        
        return results
    
    def predict_video(self, video_path, frame_interval=10, max_frames=50):
        """Predict if a video is real or fake"""
        print(f"\n🎬 Processing video: {video_path}")
        print("="*60)
        
        # Check if file exists
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
        
        # Extract frames
        frames = self.extract_frames(video_path, frame_interval, max_frames)
        
        if len(frames) == 0:
            return {"error": "No frames extracted from video"}
        
        # Process each frame
        predictions = []
        confidences = []
        face_detected_count = 0
        
        print(f"\n🔍 Analyzing {len(frames)} frames...")
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Detect face
            face = self.detect_faces(frame)
            
            if face is not None:
                # Predict
                label, confidence = self.predict_frame(face)
                predictions.append(label)
                confidences.append(confidence)
                face_detected_count += 1
                
                # Show progress for first few frames
                if i < 5:
                    print(f"   Frame {i}: {label} ({confidence:.2%})")
        
        if len(predictions) == 0:
            return {"error": "No faces detected in any frame"}
        
        # Majority voting
        real_count = predictions.count("REAL")
        fake_count = predictions.count("FAKE")
        
        final_label = "REAL" if real_count > fake_count else "FAKE"
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Results
        results = {
            "video_path": video_path,
            "final_prediction": final_label,
            "confidence": avg_confidence,
            "real_frames": real_count,
            "fake_frames": fake_count,
            "total_frames_processed": len(predictions),
            "frames_with_faces": face_detected_count,
            "frame_predictions": predictions,
            "frame_confidences": confidences
        }
        
        print("\n" + "="*60)
        print("📊 PREDICTION RESULTS")
        print("="*60)
        print(f"🎯 Final Prediction: {final_label}")
        print(f"📊 Confidence: {avg_confidence:.2%}")
        print(f"📈 Real frames: {real_count}/{len(predictions)}")
        print(f"📈 Fake frames: {fake_count}/{len(predictions)}")
        print(f"👤 Frames with faces: {face_detected_count}/{len(frames)}")
        
        return results

def main():
    """Main function to test the predictor"""
    
    print("="*60)
    print("🎭 DEEPFAKE DETECTION PREDICTOR")
    print("="*60)
    
    # Create predictor
    predictor = VideoPredictor()
    
    while True:
        print("\n" + "="*60)
        print("Options:")
        print("1. Predict on a video")
        print("2. Predict on an image")
        print("3. Exit")
        choice = input("\nEnter your choice (1, 2, or 3): ")
        
        if choice == '1':
            video_path = input("\n📹 Enter path to video file: ")
            results = predictor.predict_video(video_path)
            if "error" in results:
                print(f"❌ Error: {results['error']}")
            else:
                print("\n✅ Prediction complete!")
        
        elif choice == '2':
            image_path = input("\n🖼️ Enter path to image file: ")
            results = predictor.predict_image(image_path)
            if "error" in results:
                print(f"❌ Error: {results['error']}")
            else:
                print("\n✅ Prediction complete!")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()