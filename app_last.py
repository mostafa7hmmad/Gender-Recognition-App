import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
from datetime import datetime

# 0️⃣ Google Drive Upload Function
def upload_to_drive(image_bytes, filename, folder_id):
    try:
        # Create Drive API client
        creds = service_account.Credentials.from_service_account_file(
            'service-account.json',
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)

        # Create file metadata and media upload
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        media = MediaIoBaseUpload(io.BytesIO(image_bytes), 
                                mimetype='image/jpeg')

        # Execute upload
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return True
    except Exception as e:
        st.error(f"Drive Upload Error: {str(e)}")
        return False

# 1️⃣ Load Model and Face Detector
@st.cache_resource
def load_model_and_classes():
    model = load_model("Model_2.h5")
    class_idx = np.load("class_indices.npy", allow_pickle=True).item()
    index_to_class = {v: k for k, v in class_idx.items()}
    return model, index_to_class

@st.cache_resource
def load_face_detector():
    return MTCNN()

model, index_to_class = load_model_and_classes()
detector = load_face_detector()
min_face_width = 36

# 2️⃣ App Interface
st.title("🎭 Automated-Gender-Classification-Using-Facial-Recognition")
st.write("Upload an image or use webcam. Images are automatically saved to Google Drive.")

# 3️⃣ Image Input Handling
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Or capture an image using your webcam")
image_source = camera_image or uploaded_file

if image_source:
    # 4️⃣ Save Original Image to Google Drive
    try:
        image_bytes = image_source.getvalue()
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        if upload_to_drive(image_bytes, filename, "1N7PmIx1hHBlXtYNvi0u2dYK18n5wWtPA"):
            st.success("✅ Image saved to Google Drive successfully!")
    except Exception as upload_error:
        st.error(f"⚠️ Failed to save image: {str(upload_error)}")

    # 5️⃣ Image Processing
    image = Image.open(image_source).convert("RGB")
    img_array = np.array(image)
    faces = detector.detect_faces(img_array)

    if not faces:
        st.error("🚨 No faces detected. Please try another image.")
    else:
        output_img = cv2.cvtColor(img_array.copy(), cv2.COLOR_RGB2BGR)
        results = []

        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            if w < min_face_width: continue

            # Face processing and prediction
            face_region = img_array[y:y+h, x:x+w]
            face_input = cv2.resize(face_region, (64, 64)).astype('float32') / 255
            prediction = model.predict(np.expand_dims(face_input, axis=0))[0][0]
            
            # Generate results
            label_index = int(prediction >= 0.5)
            confidence = prediction if label_index else 1 - prediction
            label = index_to_class[label_index]
            results.append((label, confidence * 100, (x, y, x+w, y+h)))

            # Draw annotations
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output_img, 
                       f"{label} ({confidence*100:.1f}%)", 
                       (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show results
        if results:
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), 
                    caption="🔍 Detection Results", 
                    use_container_width=True)
            for i, (label, conf, _) in enumerate(results, 1):
                st.write(f"**👤 Face {i}:** {label} ({conf:.2f}% confidence)")
        else:
            st.warning("⚠️ No faces met minimum size requirements")
else:
    st.info("📢 Please upload or capture an image to begin.")
