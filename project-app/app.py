import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("YOLOv8 Object Detection Web App")

# Load model
model = YOLO("yolov8m.pt")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # YOLO detection
    results = model(img)
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Objects", use_column_width=True)