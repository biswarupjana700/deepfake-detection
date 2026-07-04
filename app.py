import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import sys
import time
import tempfile

# Add src to path
sys.path.append('src')
from model import get_model

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="VERITAS AI | DeepFake Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# ============ GLOBAL PREMIUM THEME (CSS) — GitHub-dark inspired: navy + violet ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background — GitHub-dark navy, subtle violet glow in the corners */
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(163,113,247,0.16) 0%, transparent 45%),
        radial-gradient(circle at 95% 15%, rgba(31,111,235,0.14) 0%, transparent 40%),
        #0d1117;
    color: #c9d1d9;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #010409;
    border-right: 1px solid #21262d;
}

p, span, div, li { color: #c9d1d9; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 9px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #388bfd, #a371f7); border-radius: 10px; }

/* File uploader restyle */
[data-testid="stFileUploader"] { background: transparent; }
[data-testid="stFileUploaderDropzone"] {
    background: #161b22 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #a371f7 !important;
}

/* ===== Buttons: curved, raised, "pointing outward" clickable feel ===== */
.stButton>button, [data-testid="stFileUploader"] button, [data-testid="baseButton-secondary"] {
    background: linear-gradient(180deg, #2f81f7 0%, #1f6feb 55%, #1a5fd0 100%);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid #388bfd;
    border-radius: 999px;
    padding: 0.55rem 1.4rem;
    box-shadow:
        0 4px 0 0 #133b8a,
        0 8px 16px rgba(31, 111, 235, 0.45),
        inset 0 1px 0 rgba(255,255,255,0.25);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 6px 0 0 #133b8a,
        0 12px 22px rgba(163, 113, 247, 0.45),
        inset 0 1px 0 rgba(255,255,255,0.3);
    border-color: #a371f7;
}
.stButton>button:active {
    transform: translateY(1px);
    box-shadow:
        0 2px 0 0 #133b8a,
        0 4px 10px rgba(31, 111, 235, 0.4);
}

/* Expander */
.streamlit-expanderHeader {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    color: #c9d1d9 !important;
}

/* Progress bar accent */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #1f6feb, #a371f7);
}

/* Table */
[data-testid="stTable"] {
    background: #161b22;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Check if model file exists
        model_path = './models/best_model_improved.pth'
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None
            
        model = get_model('resnet18', num_classes=2, freeze=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# ============ TRANSFORMS ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)

    confidence = probs[0][pred.item()].item()
    label = "REAL" if pred.item() == 0 else "FAKE"
    return label, confidence

# ============ VIDEO PROCESSING ============
def process_video(video_path, frame_interval=10, max_frames=50):
    """Process video and return predictions for each frame"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    if len(frames) == 0:
        return None, None, None

    # Process each frame
    predictions = []
    confidences = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, frame in enumerate(frames):
        status_text.text(f"📹 Processing frame {i+1}/{len(frames)}...")
        progress_bar.progress((i + 1) / len(frames))

        # Convert frame to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Predict
        label, confidence = predict_image(pil_image)
        predictions.append(label)
        confidences.append(confidence)

    progress_bar.empty()
    status_text.empty()

    return predictions, confidences, frames

# ============ HERO / TITLE ============
st.markdown("""
<div style="background: linear-gradient(160deg, #161b22 0%, #0d1117 60%, #010409 100%);
            border: 1px solid #30363d;
            padding: 2.8rem 2rem; border-radius: 22px; margin-bottom: 2rem; text-align: center;
            box-shadow: 0 10px 45px rgba(0,0,0,0.55), inset 0 1px 0 #161b22;">
    <div style="display:inline-flex; align-items:center; gap:0.6rem; background: linear-gradient(180deg, #1f6feb 0%, #1a5fd0 100%);
                border: 1px solid #388bfd; padding: 0.4rem 1.2rem; border-radius: 999px; margin-bottom: 1.2rem;
                box-shadow: 0 3px 0 0 #133b8a, 0 6px 14px rgba(31,111,235,0.4);">
        <span style="color:#ffffff; font-size:0.78rem; letter-spacing: 2px; font-weight:700;">AI-POWERED MEDIA SECURITY</span>
    </div>
    <h1 style="background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 30%, #a371f7 65%, #d2a8ff 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 3.5rem; font-weight: 700; margin: 0; font-family:'Space Grotesk',sans-serif; letter-spacing: -1px;">
        🛡️ VERITAS AI
    </h1>
    <p style="color: #8b949e; font-size: 1.15rem; margin-top: 0.7rem; font-weight: 500;">
        The DeepFake Sentinel — Precision Media Authenticity Verification
    </p>
</div>
""", unsafe_allow_html=True)

# ============ MODEL STATUS ============
if model:
    st.markdown("""
    <div style="background: #0d1c14; border: 1px solid #2ea043;
                border-radius: 999px; padding: 0.8rem 1.2rem; margin-bottom: 1.5rem; text-align:center;
                box-shadow: 0 3px 0 0 #163b23, 0 6px 14px rgba(46,160,67,0.25);">
        <span style="color:#3fb950; font-weight:600;">✅ Engine Online &nbsp;•&nbsp; ResNet18 Core &nbsp;•&nbsp; 81.38% Verified Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%); padding: 1.1rem; border-radius: 14px; text-align: center; margin-bottom: 1.2rem;">
        <h3 style="color: #0d1117; margin: 0; font-family:'Space Grotesk',sans-serif;">🛡️ VERITAS AI</h3>
        <p style="color: rgba(255,255,255,0.85); margin:0; font-size:0.75rem; font-weight:600;">Model Intelligence Panel</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #161b22; padding: 1rem; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 1rem;">
        <h4 style="color: #f0f0f0; margin: 0 0 0.5rem 0;">🤖 Architecture</h4>
        <p style="font-size: 1.1rem; font-weight: bold; color: #58a6ff; margin: 0;">ResNet18</p>
        <p style="color: #9aa0c9; font-size: 0.9rem;">Transfer Learning • ImageNet</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #161b22; padding: 1rem; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 1rem;">
        <h4 style="color: #f0f0f0; margin: 0 0 0.5rem 0;">📊 Performance</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div style="background: rgba(88,166,255,0.08); padding: 0.8rem; border-radius: 10px; text-align: center; border: 1px solid #30363d;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #58a6ff;">81.4%</div>
                <div style="font-size: 0.8rem; color: #9aa0c9;">Accuracy</div>
            </div>
            <div style="background: rgba(108,92,231,0.1); padding: 0.8rem; border-radius: 10px; text-align: center; border: 1px solid rgba(108,92,231,0.2);">
                <div style="font-size: 1.5rem; font-weight: bold; color: #a78bfa;">0.80</div>
                <div style="font-size: 0.8rem; color: #9aa0c9;">F1 Score</div>
            </div>
            <div style="background: rgba(0,184,148,0.08); padding: 0.8rem; border-radius: 10px; text-align: center; border: 1px solid rgba(0,184,148,0.2);">
                <div style="font-size: 1.5rem; font-weight: bold; color: #00e5b0;">86.5%</div>
                <div style="font-size: 0.8rem; color: #9aa0c9;">Precision</div>
            </div>
            <div style="background: rgba(225,112,85,0.1); padding: 0.8rem; border-radius: 10px; text-align: center; border: 1px solid rgba(225,112,85,0.2);">
                <div style="font-size: 1.5rem; font-weight: bold; color: #ff9776;">74.4%</div>
                <div style="font-size: 0.8rem; color: #9aa0c9;">Recall</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #161b22; padding: 1rem; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 1rem;">
        <h4 style="color: #f0f0f0; margin: 0 0 0.5rem 0;">📋 Test Results</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
            <span style="background: linear-gradient(135deg,#00b894,#00cec9); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight:600;">✅ Real: 88.4%</span>
            <span style="background: linear-gradient(135deg,#e17055,#d63031); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight:600;">❌ Fake: 74.4%</span>
            <span style="background: linear-gradient(135deg,#58a6ff,#a371f7); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.78rem; font-weight:600;">📊 AUC: 0.8945</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #161b22; padding: 1rem; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 1rem;">
        <h4 style="color: #f0f0f0; margin: 0 0 0.5rem 0;">📝 How It Works</h4>
        <ol style="color: #b7bce0; font-size: 0.9rem; padding-left: 1.2rem;">
            <li>Upload an image or video</li>
            <li>VERITAS analyzes the content</li>
            <li>Returns REAL or FAKE verdict</li>
            <li>Displays confidence & breakdown</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(255,193,7,0.08); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,193,7,0.35); margin-bottom: 1rem;">
        <h4 style="color: #ffd166; margin: 0 0 0.3rem 0;">⚠️ Disclaimer</h4>
        <p style="color: #e8d9a8; font-size: 0.8rem; margin: 0;">For research purposes only. Not for production or forensic/legal use.</p>
    </div>
    """, unsafe_allow_html=True)

# ============ UPLOAD SECTION ============
st.markdown("""
<div style="border: 1.5px dashed #388bfd; border-radius: 18px; padding: 2.2rem; text-align: center;
            background: rgba(88,166,255,0.03); margin: 1rem 0;">
    <div style="font-size: 3rem;">📤</div>
    <h3 style="color: #f0f0f0; font-family:'Space Grotesk',sans-serif;">Upload Media for Verification</h3>
    <p style="color: #9aa0c9;">Drag & drop or click to browse</p>
    <p style="color: #6c7299; font-size: 0.8rem;">Supported: JPG, PNG, MP4, AVI, MOV • Max 200MB</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a file...",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
    label_visibility="collapsed"
)

# ============ PROCESS UPLOAD ============
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    st.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)

    # ======== IMAGE PROCESSING ========
    if file_ext in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption=f"📎 {uploaded_file.name}", use_column_width=True)

        with col2:
            with st.spinner("🧠 Analyzing image..."):
                time.sleep(0.3)
                label, confidence = predict_image(image)

            if label == "REAL":
                st.markdown(f"""
                <div style="background: linear-gradient(180deg, #2ea043 0%, #238636 55%, #196c2c 100%); padding: 2.4rem; border-radius: 22px;
                            text-align: center; border: 1px solid #3fb950;
                            box-shadow: 0 6px 0 0 #14401d, 0 14px 30px rgba(46,160,67,0.4), inset 0 1px 0 rgba(255,255,255,0.2);">
                    <h1 style="color: white; font-size: 3rem; margin: 0; font-family:'Space Grotesk',sans-serif; letter-spacing:1px;">✅ {label}</h1>
                    <p style="color: rgba(255,255,255,0.95); font-size: 1.15rem; margin-top:0.4rem;">This image appears to be authentic</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(180deg, #f85149 0%, #da3633 55%, #a12622 100%); padding: 2.4rem; border-radius: 22px;
                            text-align: center; border: 1px solid #f85149;
                            box-shadow: 0 6px 0 0 #5c1210, 0 14px 30px rgba(218,54,51,0.4), inset 0 1px 0 rgba(255,255,255,0.2);">
                    <h1 style="color: white; font-size: 3rem; margin: 0; font-family:'Space Grotesk',sans-serif; letter-spacing:1px;">❌ {label}</h1>
                    <p style="color: rgba(255,255,255,0.95); font-size: 1.15rem; margin-top:0.4rem;">This image appears to be manipulated</p>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown(f"""
            <div style="margin-top: 1.3rem;">
                <div style="display: flex; justify-content: space-between; font-weight: 600; color:#f0f0f0;">
                    <span>Confidence Level</span>
                    <span style="color: #58a6ff;">{confidence:.1%}</span>
                </div>
                <div style="background: rgba(255,255,255,0.06); border-radius: 12px; padding: 0.25rem; overflow: hidden; border: 1px solid #30363d;">
                    <div style="background: linear-gradient(90deg, #58a6ff, #a371f7); width: {confidence*100}%; padding: 0.5rem; border-radius: 9px; text-align: center; color: #0d1117; font-weight: 800; font-size: 0.8rem;">
                        {confidence:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Details
            st.markdown(f"""
            <div style="background: #161b22; padding: 1.1rem; border-radius: 12px; margin-top: 1.1rem; border: 1px solid #30363d;">
                <p style="font-weight: 700; color: #f0f0f0; margin: 0 0 0.5rem 0;">📊 Prediction Details</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; font-size: 0.9rem; color:#b7bce0;">
                    <div><span style="color: #9aa0c9;">Prediction:</span> <b style="color:#f0f0f0;">{label}</b></div>
                    <div><span style="color: #9aa0c9;">Confidence:</span> <b style="color:#f0f0f0;">{confidence:.1%}</b></div>
                    <div><span style="color: #9aa0c9;">Model:</span> <b style="color:#f0f0f0;">ResNet18</b></div>
                    <div><span style="color: #9aa0c9;">Accuracy:</span> <b style="color:#f0f0f0;">81.38%</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ======== VIDEO PROCESSING ========
    elif file_ext in ['mp4', 'avi', 'mov']:
        st.info("🎬 Processing video... This may take a few moments.")

        # Save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            video_path = tmp_file.name

        # Display video
        st.video(video_path)

        # Process video
        with st.spinner("🧠 Analyzing video frames..."):
            predictions, confidences, frames = process_video(video_path, frame_interval=10, max_frames=50)

        # Clean up temp file
        os.unlink(video_path)

        if predictions is None:
            st.error("❌ No frames could be extracted from the video.")
        else:
            # Calculate results
            real_count = predictions.count("REAL")
            fake_count = predictions.count("FAKE")
            total_frames = len(predictions)
            avg_confidence = np.mean(confidences) if confidences else 0

            final_label = "REAL" if real_count > fake_count else "FAKE"

            # Display results
            col1, col2 = st.columns([1, 1])

            with col1:
                if final_label == "REAL":
                    st.markdown(f"""
                    <div style="background: linear-gradient(180deg, #2ea043 0%, #238636 55%, #196c2c 100%); padding: 2.4rem; border-radius: 22px;
                                text-align: center; border: 1px solid #3fb950;
                                box-shadow: 0 6px 0 0 #14401d, 0 14px 30px rgba(46,160,67,0.4), inset 0 1px 0 rgba(255,255,255,0.2);">
                        <h1 style="color: white; font-size: 3rem; margin: 0; font-family:'Space Grotesk',sans-serif; letter-spacing:1px;">✅ {final_label}</h1>
                        <p style="color: rgba(255,255,255,0.95); font-size: 1.15rem; margin-top:0.4rem;">Video appears to be authentic</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(180deg, #f85149 0%, #da3633 55%, #a12622 100%); padding: 2.4rem; border-radius: 22px;
                                text-align: center; border: 1px solid #f85149;
                                box-shadow: 0 6px 0 0 #5c1210, 0 14px 30px rgba(218,54,51,0.4), inset 0 1px 0 rgba(255,255,255,0.2);">
                        <h1 style="color: white; font-size: 3rem; margin: 0; font-family:'Space Grotesk',sans-serif; letter-spacing:1px;">❌ {final_label}</h1>
                        <p style="color: rgba(255,255,255,0.95); font-size: 1.15rem; margin-top:0.4rem;">Video appears to be manipulated</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="background: #161b22; padding: 1.6rem; border-radius: 18px; border: 1px solid #30363d; height: 100%;">
                    <h4 style="color: #f0f0f0; margin: 0 0 1rem 0;">📊 Video Analysis Summary</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;">
                        <div style="background: rgba(88,166,255,0.06); padding: 0.9rem; border-radius: 10px; text-align: center; border: 1px solid #30363d;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #58a6ff;">{total_frames}</div>
                            <div style="font-size: 0.8rem; color: #9aa0c9;">Frames Analyzed</div>
                        </div>
                        <div style="background: rgba(0,184,148,0.08); padding: 0.9rem; border-radius: 10px; text-align: center; border: 1px solid rgba(0,184,148,0.2);">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #00e5b0;">{real_count}</div>
                            <div style="font-size: 0.8rem; color: #9aa0c9;">Real Frames</div>
                        </div>
                        <div style="background: rgba(225,112,85,0.1); padding: 0.9rem; border-radius: 10px; text-align: center; border: 1px solid rgba(225,112,85,0.2);">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #ff9776;">{fake_count}</div>
                            <div style="font-size: 0.8rem; color: #9aa0c9;">Fake Frames</div>
                        </div>
                        <div style="background: rgba(108,92,231,0.1); padding: 0.9rem; border-radius: 10px; text-align: center; border: 1px solid rgba(108,92,231,0.2);">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #a78bfa;">{avg_confidence:.1%}</div>
                            <div style="font-size: 0.8rem; color: #9aa0c9;">Avg Confidence</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar for video
            st.markdown(f"""
            <div style="margin-top: 1.3rem;">
                <div style="display: flex; justify-content: space-between; font-weight: 600; color:#f0f0f0;">
                    <span>Overall Confidence</span>
                    <span style="color: #58a6ff;">{avg_confidence:.1%}</span>
                </div>
                <div style="background: rgba(255,255,255,0.06); border-radius: 12px; padding: 0.25rem; overflow: hidden; border: 1px solid #30363d;">
                    <div style="background: linear-gradient(90deg, #58a6ff, #a371f7); width: {avg_confidence*100}%; padding: 0.5rem; border-radius: 9px; text-align: center; color: #0d1117; font-weight: 800; font-size: 0.8rem;">
                        {avg_confidence:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show frame-by-frame breakdown
            with st.expander("📋 View Frame-by-Frame Analysis"):
                frame_data = []
                for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                    frame_data.append({
                        "Frame": i + 1,
                        "Prediction": pred,
                        "Confidence": f"{conf:.1%}"
                    })
                st.table(frame_data)

# ============ FOOTER ============
st.markdown("""
<div style="text-align: center; padding: 2.2rem 0 0.5rem 0; color: #6c7299; font-size: 0.8rem;
            border-top: 1px solid #30363d; margin-top: 2.5rem;">
    🛡️ <b style="color:#58a6ff;">VERITAS AI</b> — DeepFake Sentinel &nbsp;•&nbsp; Streamlit • PyTorch • ResNet18 &nbsp;•&nbsp; v2.0 Premium
</div>
""", unsafe_allow_html=True)