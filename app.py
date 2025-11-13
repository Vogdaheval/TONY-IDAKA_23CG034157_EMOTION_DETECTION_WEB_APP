# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# -------------------------------------------------
# 1. Page config (nice title + icon)
# -------------------------------------------------
st.set_page_config(
    page_title="Emotion Detection Web App",
    page_icon="😃",
    layout="centered"
)

# -------------------------------------------------
# 2. Title & short description
# -------------------------------------------------
st.title("😃 Real-time Facial Emotion Detection")
st.markdown(
    """
    Upload a **clear photo of a single face** and the model will predict one of the seven emotions:  
    **Angry 😡 · Disgust 🤢 · Fear 😱 · Happy 😀 · Neutral 😐 · Sad 😢 · Surprise 😲**
    """
)

# -------------------------------------------------
# 3. Load the pre-trained model (cached)
# -------------------------------------------------
@st.cache_resource
def load_emotion_model():
    model_path = "model.h5"          # <-- put your .h5 file in the repo root
    if not os.path.exists(model_path):
        st.error(
            f"Model file `{model_path}` not found! "
            "Please add it to the repository root."
        )
        st.stop()
    try:
        # tf 2.16+ works with the same load_model call
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_emotion_model()

# -------------------------------------------------
# 4. Emotion label mapping (order must match training)
# -------------------------------------------------
emotion_labels = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

# -------------------------------------------------
# 5. Image upload widget
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # -------------------------------------------------
    # 6. Open with PIL (keeps alpha channel handling safe)
    # -------------------------------------------------
    image = Image.open(uploaded_file)

    # Convert to RGB if it has an alpha channel (PNG)
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------------------------
    # 7. Pre-process for the model
    # -------------------------------------------------
    # Most FER models expect 48×48 or 224×224 grayscale – adjust size below
    TARGET_SIZE = (48, 48)          # <-- change if your model uses 224×224

    # Resize → grayscale → normalize
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE)
    normalized = resized / 255.0
    final_input = normalized.reshape(1, TARGET_SIZE[0], TARGET_SIZE[1], 1)

    # -------------------------------------------------
    # 8. Predict
    # -------------------------------------------------
    with st.spinner("Analyzing emotion..."):
        preds = model.predict(final_input)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(preds[0][pred_idx])
        emotion = emotion_labels[pred_idx]

    # -------------------------------------------------
    # 9. Show result
    # -------------------------------------------------
    st.subheader("Prediction")
    st.markdown(f"**Emotion:** {emotion}  ")
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.2%}")

    # Optional: show all probabilities in an expander
    with st.expander("See all probabilities"):
        prob_df = {
            "Emotion": emotion_labels,
            "Probability": [float(p) for p in preds[0]]
        }
        st.dataframe(prob_df, use_container_width=True)

else:
    st.info("👆 Upload an image to get started.")
