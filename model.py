# model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

# Define path to model (will load a pretrained one if not found)
MODEL_PATH = "emotion_model.h5"

def load_emotion_model():
    """
    Loads the pretrained emotion detection model.
    If the file doesn't exist, downloads one automatically.
    """
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading pretrained model...")
        model_url = "https://github.com/omar178/Emotion-recognition/releases/download/v1.0/emotion_model.h5"
        tf.keras.utils.get_file(MODEL_PATH, origin=model_url)
        print("✅ Model downloaded successfully!")

    model = load_model(MODEL_PATH)
    print("✅ Model loaded and ready.")
    return model


def preprocess_image(image):
    """
    Converts an image to the format the model expects.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face


def predict_emotion(model, image):
    """
    Predict emotion from an image using the loaded model.
    """
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]
    label = emotions[np.argmax(predictions)]
    confidence = np.max(predictions)
    return label, confidence
