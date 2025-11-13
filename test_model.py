# test_model.py
from model import load_emotion_model, predict_emotion
import cv2

# Load model
model = load_emotion_model()

# Load a sample face image for testing
# You can replace this with any image path later
sample_image_path = "test_face.jpg"

# Check if file exists
try:
    image = cv2.imread(sample_image_path)
    if image is None:
        print("⚠️ No test image found. Please add a small face image as 'test_face.jpg' in your project folder.")
    else:
        label, confidence = predict_emotion(model, image)
        print(f"Predicted Emotion: {label} ({confidence:.2f})")
except Exception as e:
    print(f"❌ Error: {e}")
