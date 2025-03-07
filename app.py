import streamlit as st
import numpy as np
import cv2
import gdown
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# 1️⃣ **Download and Load Model from Google Drive and Face Detector Efficiently**
model_url = "https://drive.google.com/uc?id=1yV06WrDAoUbZGKTDIw45D3uP3C3ZxNJB"

@st.cache_resource
def load_model_and_classes():
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
st.write("Upload an image, capture from webcam, or use live face detection.")
st.write("The app will draw a bounding box around detected faces and predict gender with confidence.")

# 3️⃣ **User Input: Upload, Capture Image, or Live Camera**
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Or capture an image using your webcam")
use_live_cam = st.checkbox("📹 Use Live Face Detection")

# Determine the source of the image
image_source = camera_image if camera_image is not None else uploaded_file

if use_live_cam:
    st.write("Starting Live Camera... Press 'Stop' to end.")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("🚨 Failed to grab frame. Ensure your webcam is working.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)

        for face in faces:
            x, y, width, height = face["box"]
            if width < min_face_width:
                continue
            
            x, y, x2, y2 = max(0, x), max(0, y), x + width, y + height
            face_region = frame_rgb[y:y2, x:x2]
            face_resized = cv2.resize(face_region, (64, 64))
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            prediction = model.predict(face_input)[0][0]
            label_index = 1 if prediction >= 0.5 else 0
            confidence = prediction if label_index == 1 else 1 - prediction
            label = index_to_class[label_index]
            confidence_percent = confidence * 100

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} ({confidence_percent:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR")
    
    cap.release()
    cv2.destroyAllWindows()

elif image_source:
    image = Image.open(image_source).convert("RGB")
    img_array = np.array(image)

    faces = detector.detect_faces(img_array)

    if not faces:
        st.error("🚨 No faces detected. Please upload a clear image with visible faces.")
    else:
        output_img = img_array.copy()
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        results = []
        for i, face in enumerate(faces):
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
