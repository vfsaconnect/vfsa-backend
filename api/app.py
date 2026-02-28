from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os
import cv2
import tensorflow as tf

# =====================================================
# INITIALIZE FLASK APP
# =====================================================

app = Flask(__name__, static_folder="static")
CORS(app)

# =====================================================
# LOAD MODELS (Railway Safe Paths)
# =====================================================

# Get the directory where this file (app.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to project root (since app.py is in backend folder)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# AI model directory is at project root/ai_model
AI_MODEL_DIR = os.path.join(PROJECT_ROOT, "ai_model")

# Model paths
SVM_PATH = os.path.join(AI_MODEL_DIR, "isl_landmark_model.pkl")
SCALER_PATH = os.path.join(AI_MODEL_DIR, "scaler.pkl")
CNN_PATH = os.path.join(AI_MODEL_DIR, "cnn_model.h5")

print("📂 BASE_DIR:", BASE_DIR)
print("📂 PROJECT_ROOT:", PROJECT_ROOT)
print("📂 AI_MODEL_DIR:", AI_MODEL_DIR)
print("📂 SVM_PATH:", SVM_PATH)
print("📂 SCALER_PATH:", SCALER_PATH)
print("📂 CNN_PATH:", CNN_PATH)

svm_model = None
scaler = None
cnn_model = None

# Try loading SVM model
try:
    if os.path.exists(SVM_PATH):
        svm_model = joblib.load(SVM_PATH)
        print("✅ SVM model loaded")
    else:
        print("⚠️ SVM model not found at:", SVM_PATH)
except Exception as e:
    print("❌ SVM load error:", e)

# Try loading scaler
try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler loaded")
    else:
        print("⚠️ Scaler not found at:", SCALER_PATH)
except Exception as e:
    print("❌ Scaler load error:", e)

# Try loading CNN model
try:
    if os.path.exists(CNN_PATH):
        cnn_model = tf.keras.models.load_model(CNN_PATH)
        print("✅ CNN model loaded")
    else:
        print("⚠️ CNN model not found at:", CNN_PATH)
except Exception as e:
    print("❌ CNN load error:", e)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("🚀 Hybrid Backend Ready")
print("=" * 50)

# =====================================================
# API ROUTES
# =====================================================

@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    try:
        if svm_model is None or scaler is None:
            return jsonify({"error": "SVM model not loaded"}), 500

        data = request.get_json()

        if not data or "landmarks" not in data:
            return jsonify({"error": "No landmarks provided"}), 400

        landmarks = np.array(data["landmarks"])

        if len(landmarks) == 63:
            landmarks = np.concatenate([landmarks, np.zeros(63)])

        if len(landmarks) != 126:
            return jsonify({"error": f"Expected 126 features, got {len(landmarks)}"}), 400

        landmarks = landmarks.reshape(1, -1)
        landmarks = scaler.transform(landmarks)

        pred = svm_model.predict(landmarks)[0]

        if hasattr(svm_model, "predict_proba"):
            conf = float(np.max(svm_model.predict_proba(landmarks)))
        else:
            conf = 1.0

        letter = LABELS[int(pred)] if isinstance(pred, (int, np.integer)) else str(pred)

        return jsonify({
            "letter": letter,
            "confidence": conf,
            "model": "SVM"
        })

    except Exception as e:
        print("❌ predict_landmarks error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        if cnn_model is None:
            return jsonify({"error": "CNN model not loaded"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        img_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

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
        print("❌ predict_image error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "svm_loaded": svm_model is not None,
        "cnn_loaded": cnn_model is not None,
        "scaler_loaded": scaler is not None,
        "paths": {
            "base_dir": BASE_DIR,
            "project_root": PROJECT_ROOT,
            "ai_model_dir": AI_MODEL_DIR
        }
    })


# =====================================================
# SERVE REACT FRONTEND (Optional - uncomment if you want to serve React)
# =====================================================

# @app.route("/", defaults={"path": ""})
# @app.route("/<path:path>")
# def serve_react(path):
#     """Serve React frontend for all non-API routes"""
#     # Skip API routes
#     if path.startswith(("predict_", "health")):
#         return jsonify({"error": "API endpoint not found"}), 404
    
#     file_path = os.path.join(app.static_folder, path)
#
#     if path != "" and os.path.exists(file_path):
#         return send_from_directory(app.static_folder, path)
#
#     return send_from_directory(app.static_folder, "index.html")


# =====================================================
# SAFE ROOT ROUTE
# =====================================================

@app.route("/")
def home():
    return jsonify({
        "message": "VFSA Backend Running",
        "health_endpoint": "/health",
        "endpoints": {
            "predict_landmarks": "/predict_landmarks (POST)",
            "predict_image": "/predict_image (POST)",
            "health": "/health (GET)"
        }
    })


# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    print(f"🚀 Server starting on port {port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📊 Health check: http://localhost:{port}/health")
    print(f"🏠 Home: http://localhost:{port}/")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=port, debug=debug)