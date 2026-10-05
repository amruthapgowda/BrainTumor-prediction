import os
import cv2
import numpy as np

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

print("Model Loaded Successfully!")
print("Open: http://127.0.0.1:5000")


def get_className(classNo):
    if classNo == 0:
        return "NO Tumor Detected"
    elif classNo == 1:
        return "Tumor Detected"


def predict_image(img_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (64, 64))

    img = img.astype('float32') / 255.0

    input_img = np.expand_dims(img, axis=0)

    prediction = model.predict(input_img, verbose=0)

    predicted_class = np.argmax(prediction, axis=1)[0]

    confidence = float(np.max(prediction) * 100)

    return predicted_class, confidence


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"})

    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    filename = secure_filename(file.filename)

    file_path = os.path.join(upload_folder, filename)

    file.save(file_path)

    predicted_class, confidence = predict_image(file_path)

    result = get_className(predicted_class)

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2)
    })


if __name__ == '__main__':
    app.run(debug=True)