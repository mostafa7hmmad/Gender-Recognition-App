import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import mediapipe as mp
import pandas as pd

st.title("Automated Gender Classification Using Facial Recognition")
st.write("Take a photo to classify gender for multiple faces.")

# Load pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model.h5")

model = load_model()

# Class names
class_names = ["Female", "Male"]

# Mediapipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Take a photo using camera
captured_image = st.camera_input("Take a photo")

results = []

if captured_image:
    img = Image.open(captured_image)
    img_cv = np.array(img.convert("RGB"))  # Convert image to RGB format
    h, w, _ = img_cv.shape

    # Detect faces using Mediapipe
    results_faces = face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

    num_male = 0
    num_female = 0

    if results_faces.detections:
        faces_list = []  # List to store extracted faces
        for detection in results_faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, bw, bh = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))
            face = img_cv[y:y+bh, x:x+bw]  # Extract face from image

            # Check if face is within image bounds
            if face.shape[0] > 0 and face.shape[1] > 0:
                face_pil = Image.fromarray(face).resize((64, 64))  # Resize to match model input
                face_array = np.array(face_pil)
                if face_array.shape[-1] == 4:  # Convert RGBA to RGB
                    face_array = face_array[..., :3]
                face_array = face_array.astype("float32") / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                # Store face for processing
                faces_list.append((face_array, x, y, bw, bh))

        # Predict gender for each face
        for face_array, x, y, bw, bh in faces_list:
            prediction = model.predict(face_array)
            predicted_label = "Male" if prediction[0][0] >= 0.5 else "Female"

            # Count males and females
            if predicted_label == "Male":
                num_male += 1
            else:
                num_female += 1

            # Draw rectangle around face with label
            cv2.rectangle(img_cv, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            label_text = f"{predicted_label}"
            cv2.putText(img_cv, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display results
    img_pil = Image.fromarray(img_cv)
    st.image(img_pil, caption=f"Detected: {num_female} Females, {num_male} Males", use_column_width=True)
    st.success(f"Number of Females: {num_female}, Number of Males: {num_male}")
