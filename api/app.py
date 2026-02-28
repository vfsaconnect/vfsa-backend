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
# LOAD MODELS (Railway Safe Absolute Paths)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
AI_MODEL_DIR = os.path.join(PROJECT_ROOT, "ai_model")

SVM_PATH = os.path.join(AI_MODEL_DIR, "isl_landmark_model.pkl")
SCALER_PATH = os.path.join(AI_MODEL_DIR, "scaler.pkl")
CNN_PATH = os.path.join(AI_MODEL_DIR, "cnn_model.h5")

print("📂 BASE_DIR:", BASE_DIR)
print("📂 PROJECT_ROOT:", PROJECT_ROOT)
print("📂 AI_MODEL_DIR:", AI_MODEL_DIR)

svm_model = None
scaler = None
cnn_model = None

try:
    if os.path.exists(SVM_PATH):
        svm_model = joblib.load(SVM_PATH)
        print("✅ SVM model loaded")
    else:
        print(f"⚠️ SVM model not found at {SVM_PATH}")
except Exception as e:
    print(f"❌ Failed loading SVM: {e}")

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler loaded")
    else:
        print(f"⚠️ Scaler not found at {SCALER_PATH}")
except Exception as e:
    print(f"❌ Failed loading Scaler: {e}")

try:
    if os.path.exists(CNN_PATH):
        cnn_model = tf.keras.models.load_model(CNN_PATH)
        print("✅ CNN model loaded")
    else:
        print(f"⚠️ CNN model not found at {CNN_PATH}")
except Exception as e:
    print(f"❌ Failed loading CNN: {e}")

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
print("🚀 Hybrid Backend Ready")

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
            return jsonify({
                "error": f"Expected 126 features, got {len(landmarks)}"
            }), 400

        landmarks = landmarks.reshape(1, -1)
        landmarks = scaler.transform(landmarks)

        pred = svm_model.predict(landmarks)[0]

        if hasattr(svm_model, "predict_proba"):
            conf = float(np.max(svm_model.predict_proba(landmarks)))
        else:
            conf = 1.0

        if isinstance(pred, (int, np.integer)):
            letter = LABELS[int(pred)]
        else:
            letter = str(pred)

        return jsonify({
            "letter": letter,
            "confidence": conf,
            "model": "SVM"
        })

    except Exception as e:
        print(f"❌ Error in predict_landmarks: {e}")
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
        print(f"❌ Error in predict_image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "svm_loaded": svm_model is not None,
        "cnn_loaded": cnn_model is not None,
        "scaler_loaded": scaler is not None
    })


# =====================================================
# SERVE REACT FRONTEND
# =====================================================

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path.startswith(("predict_", "health")):
        return jsonify({"error": "API endpoint not found"}), 404

    file_path = os.path.join(app.static_folder, path)

    if path != "" and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)

    return send_from_directory(app.static_folder, "index.html")


# =====================================================
# RUN SERVER (Local Only)
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    print(f"🚀 Server starting on port {port}")
    print(f"📁 Serving static files from: {app.static_folder}")
    print(f"🔧 Debug mode: {debug}")

    app.run(host="0.0.0.0", port=port, debug=debug)