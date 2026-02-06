# model.py
import os
from deepface import DeepFace

def predict_emotion(image_path):
    """
    Analyze an image and return the dominant emotion and all confidence scores.
    """
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False
        )

        # DeepFace can return list or dict
        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result['dominant_emotion']
        scores = result['emotion']

        # Round scores to 2 decimals
        scores = {k: round(float(v), 2) for k, v in scores.items()}

        return {
            "dominant_emotion": dominant_emotion,
            "scores": scores
        }
    except Exception as e:
        print("Error predicting emotion:", e)
        return {
            "dominant_emotion": "Error",
            "scores": {}
        }
