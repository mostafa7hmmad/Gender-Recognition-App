import streamlit as st
import numpy as np
import cv2
import joblib
import lz4  # Ensure LZ4 is installed
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
st.title("🎭 Gender Classification App")
st.write("Upload an image, use your webcam, or start a live stream to detect faces and classify gender.")
st.write("The app will draw a bounding box around detected faces and predict gender with confidence.")

# Live Stream toggle
use_live_stream = st.checkbox("Use Live Stream")

if use_live_stream:
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
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Take a photo")
    image_source = uploaded_file or camera_image

    if image_source:
        image = Image.open(image_source).convert("RGB")
        img_array = np.array(image)

        faces = detector.detect_faces(img_array)

        if not faces:
            st.error("🚨 No faces detected. Please upload a clear image with visible faces.")
        else:
            output_img = img_array.copy()
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            results = []
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

                results.append((label, confidence_percent, (x, y, x2, y2)))

                cv2.rectangle(output_img, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, f"{label}: {confidence_percent:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            st.image(output_img, caption="Detected Faces with Gender Classification", use_container_width=True)

            for i, (label, confidence, _) in enumerate(results, 1):
                st.write(f"**Face {i}: {label}** – {confidence:.2f}% confidence")
