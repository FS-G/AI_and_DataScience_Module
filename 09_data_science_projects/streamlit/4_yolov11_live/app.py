import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")  # Change to "yolov11" when available

st.title("Live Object Detection with YOLO")
st.write("Turn on the webcam and detect objects in real-time.")

# Initialize session state for the checkbox
if "checkbox" not in st.session_state:
    st.session_state["checkbox"] = False

# Start/Stop button
run = st.checkbox("Start Webcam", key="checkbox")

# Open webcam stream if checkbox is checked
if run:
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    frame_placeholder = st.empty()  # Placeholder for video output

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to access webcam.")
            break

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO inference
        results = model(frame_rgb)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Display the frame in Streamlit
        frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

        # Stop if user unchecks the box
        if not st.session_state.checkbox:
            break

    cap.release()
    cv2.destroyAllWindows()
