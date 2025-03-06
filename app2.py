import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Automated Gender Classification Using Facial Recognition")
st.write("Take a photo and classify gender.")

# تحميل الموديل المدرب مسبقًا
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model.h5")

model = load_model()

# أسماء الفئات
class_names = ["Female", "Male"]

# التقاط صورة بالكاميرا
captured_image = st.camera_input("Take a photo")

if captured_image:
    img = Image.open(captured_image)
    img = img.resize((64, 64))  # تغيير الحجم ليتناسب مع مدخلات الموديل
    
    # تحويل الصورة إلى مصفوفة ومعالجتها
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # إذا كانت RGBA نحولها إلى RGB
        img_array = img_array[..., :3]
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # تنفيذ التنبؤ
    prediction = model.predict(img_array)
    predicted_label = class_names[1] if prediction[0][0] >= 0.5 else class_names[0]
    confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
    
    # عرض النتيجة
    st.image(img, caption=f"Prediction: {predicted_label} ({confidence:.2f})", use_column_width=True)
    st.success(f"Predicted Gender: {predicted_label} with confidence {confidence:.2f}")
