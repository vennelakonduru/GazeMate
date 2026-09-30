import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp

# Import your existing gaze tracking logic
from integrated_gaze import estimate_gaze

st.title("GazeMate Mobile Web Tracking")
st.write("Grant camera access to use eye tracking on your mobile browser.")

mp_face_mesh = mp.solutions.face_mesh

class MobileGazeProcessor(VideoProcessorBase):
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
            landmarks = results.multi_face_landmarks[0].landmark
            # Call your project's custom gaze estimation function
            gaze = estimate_gaze(landmarks, img.shape[1], img.shape[0])
            
            if gaze is not None:
                # Draw gaze point feedback on the web camera feed
                cv2.circle(img, (int(gaze[0]), int(gaze[1])), 10, (0, 255, 0), -1)

        return cv2.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="gazemate-mobile", video_processor_factory=MobileGazeProcessor)