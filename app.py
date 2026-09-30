import urllib.request
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Page Configuration
st.set_page_config(
    page_title="GazeMate AI | Mobile Tracking",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling
st.markdown("""
    <style>
        .main { background-color: #0E1117; }
        .metric-card {
            background-color: #1E222D;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #2E3440;
            text-align: center;
            margin-bottom: 10px;
        }
        .title-text {
            font-size: 2.2rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #00FFA3, #00B8D9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .info-box {
            background-color: #131B26;
            border-left: 4px solid #00FFA3;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Singleton cached detector to avoid re-initializing C++ shared libraries in WebRTC threads
@st.cache_resource
def get_detector():
    req = urllib.request.urlopen(MODEL_URL)
    model_buffer = req.read()
    base_options = python.BaseOptions(model_asset_buffer=model_buffer)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

class EyeGazeProcessor(VideoProcessorBase):
    def __init__(self):
        # Retrieve the pre-initialized single instance
        self.detector = get_detector()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = self.detector.detect(mp_image)

        if result.face_landmarks:
            h, w, _ = img.shape
            for landmark in result.face_landmarks[0]:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 1, (0, 255, 163), -1)

        return frame.from_ndarray(img, format="bgr24")

# UI Layout Header
col1, col2 = st.columns([1, 4])
with col1:
    st.title("👁️")
with col2:
    st.markdown('<p class="title-text">GazeMate AI</p>', unsafe_allow_html=True)
    st.caption("Real-time Mobile Web Eye-Gaze Tracking Subsystem")

st.markdown("---")

st.markdown("""
<div class="info-box">
    <strong>📱 Quick Start Guide:</strong><br>
    1. Position your face clearly in front of your device camera.<br>
    2. Click <strong>START</strong> below and grant camera permissions.<br>
    3. Ensure good lighting for optimal landmark detection accuracy.
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ App Controls")
    st.success("MediaPipe Engine Active")
    st.info("Resolution: Standard Mobile WebRTC")
    st.markdown("---")
    st.header("💻 Desktop Version")
    st.write("Need system-wide desktop cursor control?")
    st.markdown("[👉 Download Windows Executable (.exe)](https://github.com/vennelakonduru/GazeMate/releases)")
    st.markdown("---")
    st.caption("GazeMate Project • Dual-Platform Deployment")

st.subheader("📹 Live Camera Feed")

webrtc_streamer(
    key="gazemate-mobile-stream",
    video_processor_factory=EyeGazeProcessor,
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False
    },
    async_processing=True
)

st.markdown("---")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric-card"><strong>Status</strong><br><span style="color:#00FFA3;">Ready</span></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><strong>Platform</strong><br>Mobile Web</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><strong>Backend</strong><br>MediaPipe Tasks</div>', unsafe_allow_html=True)
