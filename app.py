import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import gdown
import os

# URL to the model
MODEL_URL = "https://drive.google.com/uc?id=1yV06WrDAoUbZGKTDIw45D3uP3C3ZxNJB"
MODEL_PATH = "Model_2.h5"

# Download the model if it doesn't exist
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model and classes
@st.cache_resource
def load_model_and_classes():
    download_model()
    model = load_model(MODEL_PATH)
    class_idx = {0: 'female', 1: 'male'}  # Adjust according to your model
    return model, class_idx

@st.cache_resource
def load_face_detector():
    return MTCNN()

model, index_to_class = load_model_and_classes()
detector = load_face_detector()

min_face_width = 39

st.title("🎭 Gender Classification App")
st.write("Upload an image or use your webcam to detect faces and classify gender.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Or capture an image using your webcam")

image_source = camera_image if camera_image is not None else uploaded_file

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
            label_index = 1 if prediction >= 0.5 else 0
            confidence = prediction if label_index == 1 else 1 - prediction
            label = index_to_class[label_index]
            confidence_percent = confidence * 100

            results.append((label, confidence_percent, (x, y, x2, y2)))

            cv2.rectangle(output_img, (x, y), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} ({confidence_percent:.1f}%)"
            cv2.putText(output_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not results:
            st.warning("⚠️ Faces detected, but none met the minimum size requirement.")
        else:
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            st.image(output_img, caption="🔍 Detected Faces with Gender Classification", use_container_width=True)

            for i, (label, confidence, _) in enumerate(results, 1):
                st.write(f"**👤 Face {i}: {label}** – {confidence:.2f}% confidence")
else:
    st.info("📢 Please upload or capture an image to begin.")
