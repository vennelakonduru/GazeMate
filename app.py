import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

st.set_page_config(page_title="GazeMate Mobile", page_icon="👁️")
st.title("👁️ GazeMate Mobile Web Tracking")
st.write("Grant camera access to enable real-time eye-gaze tracking.")

# Standard MediaPipe solutions import
mp_face_mesh = mp.solutions.face_mesh

class EyeGazeProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize FaceMesh model instance
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
            h, w, _ = img.shape
            for landmark in results.multi_face_landmarks[0].landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="gazemate-mobile-stream",
    video_processor_factory=EyeGazeProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
