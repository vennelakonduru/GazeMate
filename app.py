import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp

st.title("GazeMate Mobile Web Tracking")
st.write("Grant camera access to use eye tracking on your mobile browser.")

# Explicit sub-module import to prevent attribute errors
mp_face_mesh = mp.solutions.face_mesh

class EyeGazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmark in results.multi_face_landmarks[0].landmark:
                h, w, _ = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)

        return cv2.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="gaze-mobile", video_processor_factory=EyeGazeProcessor)
