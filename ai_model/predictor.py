# predictor.py
# VFSA HYBRID ISL A-Z PRODUCTION FINAL VERSION

import os
import cv2
import time
import joblib
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SVM_PATH = os.path.join(BASE_DIR, "isl_landmark_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
CNN_PATH = os.path.join(BASE_DIR, "cnn_model.h5")
HAND_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# ==========================================================
# LOAD MODELS SAFELY
# ==========================================================

print("🔄 Loading VFSA Hybrid Models...")

try:
    svm_model = joblib.load(SVM_PATH)
    print("✅ SVM model loaded")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load SVM model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load scaler: {e}")

try:
    cnn_model = tf.keras.models.load_model(CNN_PATH)
    print("✅ CNN model loaded")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load CNN model: {e}")

# ==========================================================
# MEDIAPIPE HAND LANDMARKER SETUP
# ==========================================================

base_options = python.BaseOptions(model_asset_path=HAND_TASK_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

landmarker = vision.HandLandmarker.create_from_options(options)

print("✅ MediaPipe HandLandmarker loaded")

# ==========================================================
# LABELS A-Z
# ==========================================================

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ==========================================================
# HYBRID PREDICTION FUNCTION
# ==========================================================

def hybrid_predict(frame):
    """
    Hybrid Pipeline:
    1. MediaPipe → Landmark → SVM (fast path)
    2. If no landmarks → CNN fallback
    """

    start_time = time.time()

    if frame is None:
        return {
            "letter": "",
            "confidence": 0,
            "model": "None",
            "processing_time": 0
        }

    try:
        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        # ======================================================
        # FAST PATH → SVM USING LANDMARKS
        # ======================================================

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            # Convert landmarks to numpy
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

            # Normalize relative to wrist (important!)
            wrist = points[0]
            points = points - wrist

            # Flatten to 63 features
            features = points.flatten().reshape(1, -1)

            # Scale
            features = scaler.transform(features)

            # Predict
            pred = svm_model.predict(features)[0]

            if hasattr(svm_model, "predict_proba"):
                conf = float(np.max(svm_model.predict_proba(features)))
            else:
                conf = 1.0

            # Handle both numeric and string labels
            if isinstance(pred, (int, np.integer)):
                letter = LABELS[int(pred)]
            else:
                letter = str(pred)

            model_used = "SVM"

        # ======================================================
        # CNN FALLBACK PATH
        # ======================================================

        else:

            resized = cv2.resize(rgb, (128, 128))
            norm = resized / 255.0
            input_img = np.expand_dims(norm, axis=0)

            predictions = cnn_model.predict(input_img, verbose=0)[0]
            pred = int(np.argmax(predictions))
            conf = float(np.max(predictions))

            letter = LABELS[pred]
            model_used = "CNN"

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "letter": letter,
            "confidence": round(conf, 4),
            "model": model_used,
            "processing_time": processing_time
        }

    except Exception as e:
        return {
            "letter": "",
            "confidence": 0,
            "model": "Error",
            "processing_time": 0,
            "error": str(e)
        }