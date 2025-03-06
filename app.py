import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

st.title("Automated Gender Classification Using Facial Recognition")
st.write("Upload images or use your webcam to predict gender.")

# Function to render JavaScript animation
def render_js_animation():
    animation_code = """
    <div class="container" style="position:relative; height:300px; display:flex; justify-content:center; align-items:center;">
        <div class="flex">
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
        </div>
    </div>
    <style>
        .flex {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }

        .cube {
            position: relative;
            width: 60px;
            height: 60px;
            transform-style: preserve-3d;
            animation: rotate 4s infinite linear;
            margin: 10px;
        }

        .wall {
            position: absolute;
            width: 60px;
            height: 60px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #ddd;
        }

        .front { transform: translateZ(30px); }
        .back { transform: translateZ(-30px) rotateY(180deg); }
        .left { transform: rotateY(-90deg) translateX(-30px); transform-origin: center left; }
        .right { transform: rotateY(90deg) translateX(30px); transform-origin: center right; }
        .top { transform: rotateX(90deg) translateY(-30px); transform-origin: top center; }
        .bottom { transform: rotateX(-90deg) translateY(30px); }

        @keyframes rotate {
            0% { transform: rotateX(0deg) rotateY(0deg); }
            50% { transform: rotateX(180deg) rotateY(180deg); }
            100% { transform: rotateX(360deg) rotateY(360deg); }
        }
    </style>
    """
    st.markdown(animation_code, unsafe_allow_html=True)

# Call animation
render_js_animation()

# Load the pre-trained TensorFlow model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model.h5")

model = load_model()

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Webcam input for real-time capture
camera_image = st.camera_input("Or take a photo with your webcam")

# Collect images from uploads and camera
images_to_classify = []
if uploaded_files:
    images_to_classify.extend(uploaded_files)
if camera_image:
    images_to_classify.append(camera_image)

results = []
if images_to_classify:
    with st.spinner("Classifying images..."):
        for file in images_to_classify:
            try:
                img = Image.open(file)
            except Exception as e:
                st.error(f"Could not open file {getattr(file, 'name', 'camera_image')}: {e}")
                continue

            # Resize image
            img = img.resize((64, 64))

            # Convert to array and normalize
            img_array = np.array(img)
            if img_array.shape[-1] == 4:  # Convert RGBA to RGB
                img_array = img_array[..., :3]
            img_array = img_array.astype("float32") / 255.0

            # Expand dimensions to match model input shape
            img_array = np.expand_dims(img_array, axis=0)

            # Predict gender using model (Sigmoid output)
            prediction = model.predict(img_array)

            # Determine label based on Sigmoid output
            predicted_label = "Male" if prediction[0][0] >= 0.5 else "Female"

            # Display image and prediction
            st.image(img, caption=f"**Prediction:** {predicted_label}", use_container_width=True)
            results.append((getattr(file, "name", "camera_image"), predicted_label))

    st.success("Classification completed!")

    # Download results as CSV
    if results:
        results_df = pd.DataFrame(results, columns=["Filename", "Prediction"])
        csv_data = results_df.to_csv(index=False)
        st.download_button("Download Results as CSV", data=csv_data, file_name="classification_results.csv", mime="text/csv")

    # Call animation again to indicate completion
    render_js_animation()


