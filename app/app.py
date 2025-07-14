
import streamlit as st
import numpy as np
import cv2
import joblib
import lz4  # Ensure LZ4 is installed
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import gdown  # Import gdown to download model from Google Drive

# 1️⃣ **Load Model and Face Detector Efficiently**
@st.cache_resource
def load_model_and_classes():
    # Download the model from Google Drive
    file_id = "1yV06WrDAoUbZGKTDIw45D3uP3C3ZxNJB"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "Model_2.h5"
    gdown.download(url, output, quiet=True)
    
    model = load_model(output)  # Load the trained CNN model from downloaded file
    class_idx = np.load("class_indices.npy", allow_pickle=True).item()  # Load class labels
    index_to_class = {v: k for k, v in class_idx.items()}  # Map indices to class labels
    return model, index_to_class

@st.cache_resource
def load_face_detector():
    return MTCNN()  # Load MTCNN face detector

model, index_to_class = load_model_and_classes()
detector = load_face_detector()

# Set minimum face width threshold
min_face_width = 30

# 2️⃣ **App UI & Instructions**
st.title("🎭 Gender Classification App")
st.write("Upload an image or use your webcam to detect faces and classify gender.")
st.write("The app will draw a bounding box around detected faces and predict gender with confidence.")

# 3️⃣ **User Input: Upload or Capture Image**
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Or capture an image using your webcam")

# Determine the source of the image
image_source = camera_image if camera_image is not None else uploaded_file

if image_source:
    # Read image
    image = Image.open(image_source).convert("RGB")
    img_array = np.array(image)  # Convert to NumPy array

    # 4️⃣ **Face Detection with MTCNN**
    faces = detector.detect_faces(img_array)

    if not faces:
        st.error("🚨 No faces detected. Please upload a clear image with visible faces.")
    else:
        output_img = img_array.copy()
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)  # Convert for OpenCV

        results = []  # Store classification results
        for i, face in enumerate(faces):
            x, y, width, height = face["box"]
            
            # Skip faces smaller than minimum width
            if width < min_face_width:
                continue

            x, y, x2, y2 = max(0, x), max(0, y), x + width, y + height

            # Extract face ROI
            face_region = img_array[y:y2, x:x2]

            # 5️⃣ **Preprocessing for Model**
            face_resized = cv2.resize(face_region, (64, 64))  # Resize to model input size
            face_normalized = face_resized.astype("float32") / 255.0  # Normalize
            face_input = np.expand_dims(face_normalized, axis=0)  # Add batch dimension

            # 6️⃣ **Model Prediction**
            prediction = model.predict(face_input)[0][0]  # Sigmoid output
            label_index = 1 if prediction >= 0.5 else 0  # Binary classification
            confidence = prediction if label_index == 1 else 1 - prediction  # Adjust confidence
            label = index_to_class[label_index]
            confidence_percent = confidence * 100

            results.append((label, confidence_percent, (x, y, x2, y2)))

            # **Draw Bounding Box & Label**
            cv2.rectangle(output_img, (x, y), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} ({confidence_percent:.1f}%)"
            cv2.putText(output_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not results:
            st.warning("⚠️ Faces detected, but none met the minimum size requirement.")
        else:
            # Convert image back to RGB for Streamlit
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            st.image(output_img, caption="🔍 Detected Faces with Gender Classification",  use_container_width=True)

            # 7️⃣ **Display Classification Results**
            for i, (label, confidence, _) in enumerate(results, 1):
                st.write(f"**👤 Face {i}: {label}** – {confidence:.2f}% confidence")
else:
    st.info("📢 Please upload or capture an image to begin.")
