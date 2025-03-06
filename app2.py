import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

st.title("Automated Gender Classification Using Facial Recognition")
st.write("Use your front camera for real-time gender classification.")

# تحميل الموديل المدرب مسبقًا
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model.h5")

model = load_model()

# أسماء الفئات
class_names = ["Female", "Male"]

# تعريف كلاس لمعالجة الفيديو في الزمن الحقيقي
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # تحويل الإطار إلى numpy array

        # تحويل الصورة إلى PIL
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((64, 64))  # تغيير الحجم إلى المدخل المطلوب للموديل
        
        # تحويل الصورة إلى مصفوفة ومعالجتها
        img_array = np.array(img_pil)
        if img_array.shape[-1] == 4:  # إذا كانت RGBA نحولها إلى RGB
            img_array = img_array[..., :3]
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # تنفيذ التنبؤ
        prediction = model.predict(img_array)
        predicted_label = class_names[1] if prediction[0][0] >= 0.5 else class_names[0]
        confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]

        # إضافة النص إلى الصورة
        text = f"{predicted_label}: {confidence:.2f}"
        font_scale = 1
        thickness = 2
        color = (0, 255, 0)
        
        import cv2
        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# تشغيل البث عبر WebRTC باستخدام الكاميرا الأمامية
webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {"facingMode": "user"},  # اختيار الكاميرا الأمامية
        "audio": False  # تعطيل الميكروفون
    }
)
