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
        prob = round(prediction * 100, 2)
        
        # Display Image & Classification
        st.image(image, caption=f"{label} ({prob}%)", width=150)
        st.write(f"**{file.name} - {label} ({prob}%)**")

# **2. Live Webcam Face Detection & Classification**
st.subheader("Live Camera Classification")
use_camera = st.checkbox("Use Webcam")

if use_camera:
    # OpenCV Video Capture
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        male_count, female_count = 0, 0

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Extract face
            face = cv2.resize(face, (64, 64))  # Resize for model
            face = face / 255.0  # Normalize
            face = np.expand_dims(face, axis=0)  # Reshape for model

            # Prediction
            prediction = model.predict(face)[0][0]
            label = "Male" if prediction >= 0.5 else "Female"
            color = (255, 0, 0) if label == "Male" else (0, 0, 255)

            # Count Males & Females
            if label == "Male":
                male_count += 1
            else:
                female_count += 1

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show Video Frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Show Count Results
        st.write(f"**Male: {male_count}, Female: {female_count}**")

        # Exit Webcam on 'q' Key Press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
