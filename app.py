import streamlit as st
import numpy as np
import cv2
import joblib
import lz4  # Ensure LZ4 is installed
import requests
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials

# 1️⃣ **Download and Load Model from Google Drive**
@st.cache_resource
def load_model_from_drive():
    model_url = "https://drive.google.com/uc?export=download&id=1yV06WrDAoUbZGKTDIw45D3uP3C3ZxNJB"
    response = requests.get(model_url)
    with open("Model_2.h5", "wb") as f:
        f.write(response.content)
    model = load_model("Model_2.h5")
    return model

@st.cache_resource
def load_face_detector():
    return MTCNN()  # Load MTCNN face detector

model = load_model_from_drive()
detector = load_face_detector()

# Set minimum face width threshold
min_face_width = 39

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

    # Save the uploaded image to Google Drive
    image.save("user_image.jpg")
    drive_folder_id = "1N7PmIx1hHBlXtYNvi0u2dYK18n5wWtPA"  # Google Drive folder ID

    def upload_to_drive(file_name, folder_id):
        credentials = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", ["https://www.googleapis.com/auth/drive"])
        service = build("drive", "v3", credentials=credentials)
        file_metadata = {"name": file_name, "parents": [folder_id]}
        media = MediaFileUpload(file_name, mimetype="image/jpeg")
        service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    
    upload_to_drive("user_image.jpg", drive_folder_id)
    
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
            label = "Male" if prediction >= 0.5 else "Female"
            confidence = prediction if prediction >= 0.5 else 1 - prediction  # Adjust confidence
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
            st.image(output_img, caption="🔍 Detected Faces with Gender Classification", use_container_width=True)

            # 7️⃣ **Display Classification Results**
            for i, (label, confidence, _) in enumerate(results, 1):
                st.write(f"**👤 Face {i}: {label}** – {confidence:.2f}% confidence")
else:
    st.info("📢 Please upload or capture an image to begin.")
