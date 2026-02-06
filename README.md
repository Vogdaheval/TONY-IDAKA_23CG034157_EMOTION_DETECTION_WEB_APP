# Flask Emotion Detection Web App

This is a web application that detects emotions from images or webcam input using DeepFace and Flask. Users can upload images or use live webcam to see the dominant emotion and confidence scores.

## Features
- Image upload detection
- Live webcam detection
- Professional and clean UI with gradients
- Stores uploaded images in `static/uploads/`

## Requirements
- Python 3.10
- Flask
- DeepFace
- TensorFlow (CPU)
- OpenCV
- NumPy
- Pandas

## Installation
```bash
git clone https://github.com/E-DOLLZ/emotion_detection_web_App_with_flask.git
cd emotion_detection_web_App_with_flask
python -m venv .venv
pip install -r requirements.txt

Usage
python app.py

