import streamlit as st
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

st.title("Real-Time Gender Classification")

# Visitor tracking (simple session counter)
if 'visited' not in st.session_state:
    st.session_state.visited = True
    if 'visitor_count' not in st.session_state:
        st.session_state.visitor_count = 0
    st.session_state.visitor_count += 1
    # (For a global counter, add code to read/write a file or database here)
st.sidebar.write(f"Visitor Count: {st.session_state.visitor_count}")

# Load the model with a spinner animation
with st.spinner("Loading gender classification model..."):
    model = load_model("Model.h5")

# Choose input mode
mode = st.sidebar.selectbox("Input Source", ["Webcam", "Upload Image"])

if mode == "Webcam":
    st.subheader("Webcam Live Feed")
    start_feed = st.checkbox("Start Webcam")
    frame_placeholder = st.image([])  # Placeholder for video frames

    cap = cv2.VideoCapture(0)
    # Optionally set resolution: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640), etc.

    while start_feed:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not access webcam.")
            break

        # Preprocess frame
        frame_resized = cv2.resize(frame, (64, 64))
        frame_norm = frame_resized.astype('float32') / 255.0
        frame_input = np.expand_dims(frame_norm, axis=0)

        # Predict gender
        pred = model.predict(frame_input)
        if pred.shape[1] == 1:  # single output model
            prob = pred[0][0]
            label = "Male" if prob > 0.5 else "Female"
            confidence = prob if prob > 0.5 else 1 - prob
        else:  # two output model
            class_idx = int(np.argmax(pred))
            label = "Male" if class_idx == 1 else "Female"
            confidence = pred[0][class_idx]

        # Overlay label on frame
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show frame in app
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Throttle the loop
        time.sleep(0.1)
    cap.release()

else:  # Upload Image mode
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read and decode image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Preprocess image
        img_resized = cv2.resize(img, (64, 64))
        img_norm = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_norm, axis=0)
        # Predict gender
        pred = model.predict(img_input)
        if pred.shape[1] == 1:
            label = "Male" if pred[0][0] > 0.5 else "Female"
            conf = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]
        else:
            class_idx = int(np.argmax(pred))
            label = "Male" if class_idx == 1 else "Female"
            conf = pred[0][class_idx]
        # Display result
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Prediction: {label} ({conf*100:.1f}%)", use_column_width=True)
