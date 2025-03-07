import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load CNN Model
@st.cache_resource  # Keeps model in memory for efficiency
def load_model():
    return tf.keras.models.load_model("Model_1.h5")

model = load_model()

# Preprocessing Function (64x64 normalization)
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# **1. Image Upload and Classification**
st.title("Gender Classification with CNN")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
if uploaded_files:
    st.subheader("Results")
    for file in uploaded_files:
        image = Image.open(file)
        processed_img = preprocess_image(image)
        
        # Prediction
        prediction = model.predict(processed_img)[0][0]
        label = "Male" if prediction >= 0.5 else "Female"
        
        # Display Image & Corrected Classification Result
        st.image(image, caption=f"Classified as: {label}", width=150)
        st.write(f"**{file.name} - {label}**")  # ✅ FIXED: Shows only Male or Female

# **2. Capture Photo from Webcam for Classification**
st.subheader("Take a Photo for Classification")
captured_image = st.camera_input("Capture a photo")

if captured_image:
    st.subheader("Processing Captured Photo...")

    # Convert the captured image to a PIL Image
    image = Image.open(captured_image)

    # Convert image to OpenCV format for face detection
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert RGB -> BGR for OpenCV

    # Load Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no faces detected, show warning
    if len(faces) == 0:
        st.warning("No faces detected! Please take another photo.")
    else:
        male_count, female_count = 0, 0

        # Loop through detected faces
        for (x, y, w, h) in faces:
            face = image_cv[y:y+h, x:x+w]  # Crop face
            face = cv2.resize(face, (64, 64))  # Resize to model's input size
            face = face / 255.0  # Normalize
            face = np.expand_dims(face, axis=0)  # Reshape for model

            # Predict Gender
            prediction = model.predict(face)[0][0]
            label = "Male" if prediction >= 0.5 else "Female"

            # Count Male/Female
            if label == "Male":
                male_count += 1
            else:
                female_count += 1

            # Draw bounding box around face
            color = (255, 0, 0) if label == "Male" else (0, 0, 255)
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert back to RGB for displaying
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        st.image(image_cv, caption="Processed Image with Face Classification", use_column_width=True)

        # Show Gender Counts
        st.success(f"**Detected Males: {male_count}, Females: {female_count}**")
