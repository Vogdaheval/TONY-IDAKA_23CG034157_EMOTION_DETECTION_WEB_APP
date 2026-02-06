# app.py
from flask import Flask, render_template, request
import os
from model import predict_emotion

app = Flask(__name__)

# Save uploads inside static/uploads so Render can serve them publicly
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = None

    # Handle image upload from file selector or webcam
    if 'image' in request.files:
        file = request.files['image']
    elif 'file' in request.files:
        file = request.files['file']

    if not file or file.filename == '':
        return "No image provided"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        result = predict_emotion(filepath)
        # relative path for Flask static URL
        relative_path = f'uploads/{file.filename}'
        return render_template('result.html', result=result, image_path=relative_path)
    except Exception as e:
        return f"Error predicting emotion: {e}"

if __name__ == '__main__':
    app.run(debug=True)
