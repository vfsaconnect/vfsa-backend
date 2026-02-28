from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
AI_MODEL_DIR = os.path.join(PROJECT_ROOT, "ai_model")

SVM_PATH = os.path.join(AI_MODEL_DIR, "isl_landmark_model.pkl")
SCALER_PATH = os.path.join(AI_MODEL_DIR, "scaler.pkl")
CNN_PATH = os.path.join(AI_MODEL_DIR, "cnn_model.h5")

svm_model = None
scaler = None
cnn_model = None

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_svm():
    global svm_model, scaler
    if svm_model is None and os.path.exists(SVM_PATH):
        svm_model = joblib.load(SVM_PATH)
        print("✅ SVM loaded")
    if scaler is None and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler loaded")


def load_cnn():
    global cnn_model
    if cnn_model is None and os.path.exists(CNN_PATH):
        cnn_model = tf.keras.models.load_model(CNN_PATH)
        print("✅ CNN loaded")


@app.route("/")
def home():
    return jsonify({"status": "VFSA backend running"})


@app.route("/health")
def health():
    return jsonify({
        "svm_loaded": svm_model is not None,
        "cnn_loaded": cnn_model is not None
    })


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    load_svm()

    if svm_model is None or scaler is None:
        return jsonify({"error": "SVM not available"}), 500

    data = request.get_json()
    landmarks = np.array(data["landmarks"])

    if len(landmarks) == 63:
        landmarks = np.concatenate([landmarks, np.zeros(63)])

    landmarks = landmarks.reshape(1, -1)
    landmarks = scaler.transform(landmarks)

    pred = svm_model.predict(landmarks)[0]

    return jsonify({
        "letter": LABELS[int(pred)],
        "model": "SVM"
    })


@app.route("/predict_image", methods=["POST"])
def predict_image():
    load_cnn()

    if cnn_model is None:
        return jsonify({"error": "CNN not available"}), 500

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))
    norm = resized / 255.0
    input_img = np.expand_dims(norm, axis=0)

    predictions = cnn_model.predict(input_img, verbose=0)[0]
    pred = int(np.argmax(predictions))

    return jsonify({
        "letter": LABELS[pred],
        "model": "CNN"
    })