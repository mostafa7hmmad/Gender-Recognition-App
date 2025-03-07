import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import gdown

# 1️⃣ **Load Model and Face Detector Efficiently**
@st.cache_resource
def load_model_and_classes():
    model_url = "https://drive.google.com/uc?id=1yV06WrDAoUbZGKTDIw45D3uP3C3ZxNJB"
    model_path = "Model_2.h5"
    gdown.download(model_url, model_path, quiet=False)

    model = load_model(model_path)  # Load the trained CNN model
    class_idx = np.load("class_indices.npy", allow_pickle=True).item()  # Load class labels
    index_to_class = {v: k for k, v in class_idx.items()}  # Map indices to class labels
    return model, index_to_class

@st.cache_resource
def load_face_detector():
    return MTCNN()  # Load MTCNN face detector

model, index_to_class = load_model_and_classes()
detector = load_face_detector()

# Set minimum face width threshold
min_face_width = 39

# 2️⃣ **App UI & Instructions**
st.title("🎭 Live Gender Classification")
st.write("Real-time gender detection using your webcam. Faces are detected, and gender is classified continuously.")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
stop_button = st.button("Stop Live Stream")

while camera.isOpened() and not stop_button:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to grab frame")
        break

    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_array)

    for face in faces:
        x, y, width, height = face["box"]
        if width < min_face_width:
            continue

        x, y, x2, y2 = max(0, x), max(0, y), x + width, y + height
        face_region = img_array[y:y2, x:x2]

        face_resized = cv2.resize(face_region, (64, 64))
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        prediction = model.predict(face_input)[0][0]
        label = index_to_class[int(prediction >= 0.5)]
        confidence_percent = (prediction if prediction >= 0.5 else 1 - prediction) * 100

        cv2.rectangle(img_array, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, f"{label}: {confidence_percent:.2f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    FRAME_WINDOW.image(img_array)

camera.release()
