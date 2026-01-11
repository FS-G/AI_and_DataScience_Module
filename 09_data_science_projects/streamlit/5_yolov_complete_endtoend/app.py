import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model variants
model = YOLO("best.pt") 

st.title("YOLO Pistol Detection")
st.write("Upload an image to detect objects:")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width =True)
    st.write("Processing...")

    # Run YOLOv8 model on the image
    results = model(image)

    # Get the annotated image with bounding boxes
    annotated_image = results[0].plot()

    # Display the output image with detections
    st.image(annotated_image, caption="Detected Objects", use_container_width=True)

    # # Show raw results (optional)
    # st.write("Detection Results:")
    # st.write(results)
