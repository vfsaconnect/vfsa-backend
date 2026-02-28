from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# =========================================
# PATH SETUP
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
AI_MODEL_DIR = os.path.join(PROJECT_ROOT, "ai_model")

SVM_PATH = os.path.join(AI_MODEL_DIR, "isl_landmark_model.pkl")
SCALER_PATH = os.path.join(AI_MODEL_DIR, "scaler.pkl")
CNN_PATH = os.path.join(AI_MODEL_DIR, "cnn_model.h5")

print("📂 AI_MODEL_DIR:", AI_MODEL_DIR)

# =========================================
# LAZY LOAD MODELS (IMPORTANT)
# =========================================

svm_model = None
scaler = None
cnn_model = None

def load_svm():
    global svm_model, scaler
    if svm_model is None:
        print("🔄 Loading SVM model...")
        svm_model = joblib.load(SVM_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ SVM Loaded")

def load_cnn():
    global cnn_model
    if cnn_model is None:
        print("🔄 Loading CNN model...")
        cnn_model = tf.keras.models.load_model(CNN_PATH)
        print("✅ CNN Loaded")

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("🚀 Backend Boot Complete (Models NOT loaded yet)")
print("=" * 50)

# =========================================
# ROUTES
# =========================================

@app.route("/")
def home():
    return jsonify({
        "message": "VFSA Backend Running",
        "health": "/health"
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "svm_loaded": svm_model is not None,
        "cnn_loaded": cnn_model is not None
    })

@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    try:
        load_svm()

        data = request.get_json()
        landmarks = np.array(data["landmarks"])

        if len(landmarks) == 63:
            landmarks = np.concatenate([landmarks, np.zeros(63)])

        landmarks = landmarks.reshape(1, -1)
        landmarks = scaler.transform(landmarks)

        pred = svm_model.predict(landmarks)[0]
        conf = float(np.max(svm_model.predict_proba(landmarks)))

        return jsonify({
            "letter": LABELS[int(pred)],
            "confidence": conf,
            "model": "SVM"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        load_cnn()

        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128))
        norm = resized / 255.0
        input_img = np.expand_dims(norm, axis=0)

        predictions = cnn_model.predict(input_img, verbose=0)[0]
        pred = int(np.argmax(predictions))
        conf = float(np.max(predictions))

        return jsonify({
            "letter": LABELS[pred],
            "confidence": conf,
            "model": "CNN"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500