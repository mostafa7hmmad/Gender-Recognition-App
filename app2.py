import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Automated Gender Classification Using Facial Recognition")
st.write("Use your webcam for real-time gender classification.")

# تحميل الموديل المدرب مسبقًا
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model.h5")

model = load_model()

# أسماء الفئات
class_names = ["Female", "Male"]  # يجب أن يكون ترتيبها مطابقًا لترتيب تدريب الموديل

# التحكم في حالة تشغيل الكاميرا باستخدام session_state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# زر تشغيل/إيقاف الكاميرا
if st.button("Start Camera"):
    st.session_state.camera_active = True

if st.button("Stop Camera"):
    st.session_state.camera_active = False

# فتح كاميرا الويب في حالة التشغيل فقط
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from camera")
            break
        
        # تحويل لون الصورة من BGR إلى RGB (للتوافق مع PIL و TensorFlow)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((64, 64))  # ضبط الأبعاد حسب متطلبات الموديل
        
        # تحويل الصورة إلى مصفوفة
        img_array = np.array(img)
        if img_array.shape[-1] == 4:  # تحويل RGBA إلى RGB إن لزم الأمر
            img_array = img_array[..., :3]
        img_array = img_array.astype("float32") / 255.0  # تطبيع الصورة
        img_array = np.expand_dims(img_array, axis=0)  # إضافة بُعد إضافي
        
        # تنفيذ التنبؤ
        prediction = model.predict(img_array)
        predicted_label = class_names[1] if prediction[0][0] >= 0.5 else class_names[0]
        confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
        
        # إضافة النص إلى الإطار
        text = f"{predicted_label}: {confidence:.2f}"
        cv2.putText(frame_rgb, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # عرض الصورة في Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # التحقق مما إذا تم الضغط على زر الإيقاف
        if not st.session_state.camera_active:
            break

    cap.release()
st.write("Camera stream stopped.")
