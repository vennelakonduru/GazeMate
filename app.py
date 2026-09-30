import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import mediapipe as mp

# Primary import using the modern MediaPipe Tasks API
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    USE_TASKS_API = True
except ImportError:
    USE_TASKS_API = False

st.set_page_config(page_title="GazeMate Mobile", page_icon="👁️")
st.title("👁️ GazeMate Mobile Web Tracking")
st.write("Grant camera access to enable real-time eye-gaze tracking.")

class EyeGazeProcessor(VideoProcessorBase):
    def __init__(self):
        if USE_TASKS_API:
            # Load face mesh using MediaPipe Tasks API
            model_path = mp.tasks.components.utils.download_utils.download_model(
                "face_landmarker.task"
            )
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        else:
            self.landmarker = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.landmarker:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            if result.face_landmarks:
                h, w, _ = img.shape
                for landmark in result.face_landmarks[0]:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="gazemate-mobile-stream",
    video_processor_factory=EyeGazeProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
