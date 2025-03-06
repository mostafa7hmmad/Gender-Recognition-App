import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Automated Gender Classification Using Facial Recognition")
st.write("Take a photo and classify gender for multiple faces.")

# تحميل الموديل المدرب مسبقًا
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model.h5")

model = load_model()

# أسماء الفئات
class_names = ["Female", "Male"]

# تحميل كاشف الوجوه من OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# التقاط صورة بالكاميرا
captured_image = st.camera_input("Take a photo")

if captured_image:
    img = Image.open(captured_image)
    img_cv = np.array(img.convert("RGB"))  # تحويل الصورة إلى NumPy Array بصيغة RGB
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)  # تحويل الصورة إلى تدرج الرمادي للكشف عن الوجوه

    # كشف الوجوه في الصورة
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    num_male = 0
    num_female = 0

    for (x, y, w, h) in faces:
        face = img_cv[y:y+h, x:x+w]  # استخراج الوجه من الصورة
        face = cv2.resize(face, (64, 64))  # تغيير الحجم ليتناسب مع مدخلات الموديل
        face = face.astype("float32") / 255.0  # تطبيع الصورة
        face = np.expand_dims(face, axis=0)  # إضافة بعد جديد للمصفوفة

        # تنفيذ التنبؤ
        prediction = model.predict(face)
        predicted_label = class_names[1] if prediction[0][0] >= 0.5 else class_names[0]
        
        # عداد الذكور والإناث
        if predicted_label == "Male":
            num_male += 1
        else:
            num_female += 1

        # رسم مستطيل حول الوجه مع التصنيف
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_cv, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # عرض النتائج
    img_pil = Image.fromarray(img_cv)
    st.image(img_pil, caption=f"Detected: {num_female} Females, {num_male} Males", use_column_width=True)
    st.success(f"Number of Females: {num_female}, Number of Males: {num_male}")
