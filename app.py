# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="Smile",
    layout="centered"
)

st.title("Real-time Facial Emotion Detection")
st.markdown(
    """
    Upload a **clear photo of a single face**  
    Emotions: **Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise**
    """
)

# -------------------------------------------------
# Load model from Google Drive (cached)
# -------------------------------------------------
@st.cache_resource
def load_emotion_model():
    model_path = "model.h5"
    
    # If not already downloaded
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive... (77 MB, first time only)"):
            # REPLACE THIS ID WITH YOUR FILE ID
            file_id = "18dT_Z5glhLuB43cSql-Rmh327C49Yjfr"  # ← CHANGE THIS!
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded!")

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_emotion_model()

# -------------------------------------------------
# Labels & UI
# -------------------------------------------------
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

uploaded_file = st.file_uploader("Choose an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    TARGET_SIZE = (48, 48)  # Change to (224, 224) if needed
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE)
    normalized = resized / 255.0
    final_input = normalized.reshape(1, *TARGET_SIZE, 1)

    with st.spinner("Predicting..."):
        preds = model.predict(final_input)
        pred_idx = np.argmax(preds, axis=1)[0]
        confidence = preds[0][pred_idx]
        emotion = emotion_labels[pred_idx]

    st.success(f"**{emotion}**")
    st.progress(float(confidence))
    st.caption(f"Confidence: {confidence:.1%}")

    with st.expander("All probabilities"):
        for label, prob in zip(emotion_labels, preds[0]):
            st.write(f"{label}: {prob:.1%}")
else:
    st.info("Upload an image to begin.")
