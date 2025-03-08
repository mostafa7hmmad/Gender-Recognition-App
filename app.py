import streamlit as st
import numpy as np
import cv2
import joblib
import lz4
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import gdown
import base64
from datetime import datetime
import uuid
from github import Github

# 1️⃣ Load Model and Face Detector
@st.cache_resource
def load_model_and_classes():
    file_id = "1yV06WrDAoUbZGKTDIw45D3uP3C3ZxNJB"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "Model_2.h5"
    gdown.download(url, output, quiet=True)
    
    model = load_model(output)
    class_idx = np.load("class_indices.npy", allow_pickle=True).item()
    index_to_class = {v: k for k, v in class_idx.items()}
    return model, index_to_class

@st.cache_resource
def load_face_detector():
    return MTCNN()

model, index_to_class = load_model_and_classes()
detector = load_face_detector()

# GitHub Configuration
GITHUB_REPO = "mostafa7hmmad/Automated-Gender-Classification-Using-Facial-Recognition"
GITHUB_FOLDER = ".devcontainer/"
min_face_width = 35

# 2️⃣ App Interface
st.title("🎭 Gender Classification App")
st.write("Upload an image or use your webcam to detect faces and classify gender.")

# 3️⃣ GitHub Connection Setup
try:
    github = Github(st.secrets["github"]["token"])
    repo = github.get_repo(GITHUB_REPO)
    st.sidebar.success("🔗 Connected to GitHub successfully!")
except Exception as e:
    st.error(f"""🚨 GitHub connection failed: 
             {str(e)}
             
             Please check:
             1. Token permissions (repo access)
             2. Repository existence
             3. Network connection""")
    st.stop()

# 4️⃣ Image Input Handling
uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Capture image")

image_source = camera_image or uploaded_file

if image_source:
    try:
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        file_ext = "jpg" if image_source is camera_image else image_source.name.split(".")[-1]
        filename = f"{GITHUB_FOLDER}{timestamp}_{unique_id}.{file_ext}"

        # Prepare image
        image_bytes = image_source.getvalue()
        encoded_content = base64.b64encode(image_bytes).decode()

        # GitHub upload
        repo.create_file(
            path=filename,
            message=f"Added image {filename}",
            content=encoded_content,
            branch="main"
        )
        st.success(f"✅ Image saved to GitHub: {filename}")
        
    except Exception as e:
        st.error(f"""⚠️ Failed to save image: 
                 {str(e)}
                 
                 Common fixes:
                 1. Check repository write permissions
                 2. Verify folder exists in repo
                 3. Ensure token has 'repo' scope""")

    # Rest of your processing code remains the same...
    # [Keep all your existing image processing code here]
