from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
# import cv2  # Commented out temporarily
# import tensorflow as tf  # Commented out temporarily

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
print("📂 SVM_PATH:", SVM_PATH)
print("📂 SCALER_PATH:", SCALER_PATH)
print("📂 CNN_PATH:", CNN_PATH)

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
        if os.path.exists(SVM_PATH):
            svm_model = joblib.load(SVM_PATH)
            print("✅ SVM Loaded")
        else:
            print("❌ SVM model not found at:", SVM_PATH)
    
    if scaler is None:
        print("🔄 Loading Scaler...")
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler Loaded")
        else:
            print("❌ Scaler not found at:", SCALER_PATH)

def load_cnn():
    global cnn_model
    if cnn_model is None:
        print("🔄 Loading CNN model...")
        if os.path.exists(CNN_PATH):
            # CNN loading temporarily disabled
            print("⚠️ CNN loading is temporarily disabled")
            # cnn_model = tf.keras.models.load_model(CNN_PATH)
            # print("✅ CNN Loaded")
        else:
            print("❌ CNN model not found at:", CNN_PATH)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("🚀 Backend Boot Complete (Models NOT loaded yet)")
print("=" * 50)

# =========================================
# ROUTES
# =========================================

@app.route("/")
def home():
    return "VFSA Backend Running", 200

@app.route("/health")
def health():
    return "OK", 200

@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    try:
        load_svm()

        if svm_model is None or scaler is None:
            return jsonify({"error": "SVM model not available"}), 500

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

        return jsonify({
            "letter": LABELS[int(pred)],
            "confidence": conf,
            "model": "SVM"
        })

    except Exception as e:
        print("❌ predict_landmarks error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/predict_image", methods=["POST"])
def predict_image():
    # Temporarily disabled - returns message that CNN is not available
    return jsonify({
        "error": "CNN prediction temporarily disabled",
        "message": "Using SVM only mode"
    }), 503

# =========================================
# RUN SERVER
# =========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    print(f"🚀 Server starting on port {port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📊 Health check: http://localhost:{port}/health")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=port, debug=debug)