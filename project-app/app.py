import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.title("VisionX AI - Object Detection")

# Load YOLOv8 medium model
model = YOLO("yolov8m.pt")

# Camera input (works in browser)
img_file = st.camera_input("Take a picture")

if img_file:
    image = Image.open(img_file)
    img = np.array(image)

    results = model(img)
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Objects", use_column_width=True)