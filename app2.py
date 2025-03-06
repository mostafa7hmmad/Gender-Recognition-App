import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import mediapipe as mp
import pandas as pd

st.title("Automated Gender Classification Using Facial Recognition")
st.write("Upload images or use your webcam to classify gender for multiple faces.")

# تحميل الموديل المدرب مسبقًا
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model.h5")

model = load_model()

# أسماء الفئات
class_names = ["Female", "Male"]

# تحميل Mediapipe للكشف عن الوجوه
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# رفع الصور
uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
images_to_classify = uploaded_files if uploaded_files else []

results = []
if images_to_classify:
    with st.spinner("Classifying images..."):
        for file in images_to_classify:
            try:
                img = Image.open(file)
            except Exception as e:
                st.error(f"Could not open file {getattr(file, 'name', 'uploaded_image')}: {e}")
                continue
            
            img_cv = np.array(img.convert("RGB"))  # تحويل الصورة إلى NumPy Array بصيغة RGB
            h, w, _ = img_cv.shape
            
            # كشف الوجوه في الصورة باستخدام Mediapipe
            results_faces = face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
            
            num_male = 0
            num_female = 0
            
            if results_faces.detections:
                faces_list = []  # قائمة لتخزين الوجوه المستخرجة
                for detection in results_faces.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, bw, bh = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))
                    face = img_cv[y:y+bh, x:x+bw]  # استخراج الوجه من الصورة
                    
                    # التحقق من أن الوجه داخل حدود الصورة
                    if face.shape[0] > 0 and face.shape[1] > 0:
                        face = cv2.resize(face, (64, 64))  # تغيير الحجم ليتناسب مع مدخلات الموديل
                        face = face.astype("float32") / 255.0  # تطبيع الصورة
                        face = np.expand_dims(face, axis=0)
                        
                        # تخزين الوجه لمعالجته بعد ذلك
                        faces_list.append((face, x, y, bw, bh))
                
                # تنفيذ التنبؤ لكل وجه على حدة بعد المعالجة المسبقة
                for face, x, y, bw, bh in faces_list:
                    prediction = model.predict(face)
                    confidence = float(prediction[0][0])
                    predicted_label = class_names[1] if confidence >= 0.5 else class_names[0]
                    confidence = confidence if confidence >= 0.5 else 1 - confidence
                    
                    # عداد الذكور والإناث
                    if predicted_label == "Male":
                        num_male += 1
                    else:
                        num_female += 1
                    
                    # رسم مستطيل حول الوجه مع تصنيف وثقة
                    cv2.rectangle(img_cv, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    label_text = f"{predicted_label} ({confidence:.2f})"
                    cv2.putText(img_cv, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    results.append((getattr(file, "name", "uploaded_image"), predicted_label))
            
            # عرض الصورة مع النتائج
            img_pil = Image.fromarray(img_cv)
            st.image(img_pil, caption=f"Detected: {num_female} Females, {num_male} Males", use_column_width=True)
    
    st.success("Classification completed!")
    
    # تحميل النتائج كملف CSV
    if results:
        results_df = pd.DataFrame(results, columns=["Filename", "Prediction"])
        csv_data = results_df.to_csv(index=False)
        st.download_button("Download Results as CSV", data=csv_data, file_name="classification_results.csv", mime="text/csv")
