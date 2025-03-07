import streamlit as st
import numpy as np
import cv2
import gdown  # To download model from Google Drive
from PIL import Image, ExifTags
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# 🔽 **Download Model from Google Drive**
@st.cache_resource
def load_model_from_drive():
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace with correct FILE_ID
    output = "Model_2.h5"
    gdown.download(url, output, quiet=False)

    # Load the model
    model = load_model(output)
    class_idx = np.load("class_indices.npy", allow_pickle=True).item()
    index_to_class = {v: k for k, v in class_idx.items()}
    return model, index_to_class

def load_face_detector():
    return MTCNN()

# Load Model & Detector
model, index_to_class = load_model_from_drive()
detector = load_face_detector()

# 🔄 **Fix Image Rotation Using EXIF Data**
def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                return image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                return image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                return image.rotate(90, expand=True)
    except Exception:
        pass
    return image  # Return original if no rotation data found

# 🔄 **Fix Face Alignment Based on Eye Position**
def align_face(image, face):
    x, y, w, h = face["box"]
    keypoints = face["keypoints"]
    left_eye, right_eye = keypoints["left_eye"], keypoints["right_eye"]

    # Compute angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate image to align face correctly
    center = (x + w // 2, y + h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned_img = cv2.warpAffine(np.array(image), M, (image.width, image.height))

    return Image.fromarray(aligned_img)

# 🔍 **Face Detection**
def detect_faces(image_array):
    faces = detector.detect_faces(image_array)
    return faces

# 🤖 **Predict Gender from Face**
def predict_gender(face_region):
    face_resized = cv2.resize(face_region, (64, 64))
    face_normalized = face_resized.astype("float32") / 255.0
    face_input = np.expand_dims(face_normalized, axis=0)

    prediction = model.predict(face_input)[0][0]
    label_index = 1 if prediction >= 0.5 else 0
    confidence = prediction if label_index == 1 else 1 - prediction
    label = index_to_class[label_index]
    confidence_percent = confidence * 100
    return label, confidence_percent

# 🎥 **Live Webcam Face Detection & Classification**
def live_webcam():
    st.write("📡 **Live Webcam Face Detection & Classification**")

    video_capture = cv2.VideoCapture(0)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to access webcam!")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces(frame_rgb)

        for face in faces:
            x, y, w, h = face["box"]
            face_region = frame_rgb[y:y+h, x:x+w]

            # Fix face alignment before classification
            aligned_face = align_face(Image.fromarray(frame_rgb), face)
            face_region = np.array(aligned_face)[y:y+h, x:x+w]

            label, confidence = predict_gender(face_region)

            # Draw Bounding Box & Prediction
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label_text = f"{label} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display Streamlit Video
        st.image(frame, channels="BGR")

    video_capture.release()

# 🚀 **Streamlit UI**
st.title("🎭 Gender Classification AI (Live & Upload)")
st.write("Upload an image or use the webcam for real-time classification.")

# 📸 **Image Upload**
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Capture Image")

if uploaded_file or camera_image:
    image_source = uploaded_file if uploaded_file else camera_image
    image = Image.open(image_source).convert("RGB")
    image = correct_image_orientation(image)  # Fix rotation
    img_array = np.array(image)

    faces = detect_faces(img_array)
    
    if not faces:
        st.error("🚨 No faces detected. Try using a clearer image.")
    else:
        output_img = img_array.copy()
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        results = []
        for face in faces:
            x, y, w, h = face["box"]
            face_region = img_array[y:y+h, x:x+w]

            # Fix face alignment before classification
            aligned_face = align_face(image, face)
            face_region = np.array(aligned_face)[y:y+h, x:x+w]

            label, confidence = predict_gender(face_region)

            results.append((label, confidence, (x, y, x+w, y+h)))

            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label_text = f"{label} ({confidence:.1f}%)"
            cv2.putText(output_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        st.image(output_img, caption="🔍 Corrected and Classified Faces", use_column_width=True)

        for i, (label, confidence, _) in enumerate(results, 1):
            st.write(f"**👤 Face {i}: {label}** – {confidence:.2f}% confidence")

# 🎥 **Live Webcam Button**
if st.button("🎥 Start Live Webcam"):
    live_webcam()
