import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load pretrained model
model = load_model("models/emotion_model.h5")

# Emotion categories
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("😊 AI Emotion Detection App")
st.write("Upload an image or use your webcam to detect emotions in real-time.")

# Choose input mode
option = st.radio("Select Input Mode:", ("📸 Upload Image", "🎥 Live Webcam"))

if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        image = image.convert('L')
        image = image.resize((48, 48))
        img = img_to_array(image)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        # Predict emotion
        prediction = model.predict(img)
        label = class_labels[prediction.argmax()]

        st.success(f"Predicted Emotion: **{label}**")

elif option == "🎥 Live Webcam":
    st.warning("Press 'Start' to begin webcam detection (requires camera access).")

    run = st.checkbox("Start")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Cannot access camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
    st.info("Webcam stopped.")

