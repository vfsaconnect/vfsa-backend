import os
import cv2
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
OUTPUT_CSV = os.path.join(BASE_DIR, "isl_features_AZ.csv")

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1
)

detector = HandLandmarker.create_from_options(options)

def extract_126_features(result):
    left = np.zeros((21, 3))
    right = np.zeros((21, 3))

    if result.hand_landmarks and result.handedness:
        for lm, handed in zip(result.hand_landmarks, result.handedness):
            label = handed[0].category_name
            arr = np.array([[p.x, p.y, p.z] for p in lm])

            if label == "Left":
                left = arr
            elif label == "Right":
                right = arr

    return np.concatenate([left.flatten(), right.flatten()])

def preprocess_variants(img):
    variants = []

    variants.append(img)

    bright = cv2.convertScaleAbs(img, alpha=1.4, beta=40)
    variants.append(bright)

    contrast = cv2.convertScaleAbs(img, alpha=1.6, beta=20)
    variants.append(contrast)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    variants.append(gray)

    blur = cv2.GaussianBlur(img, (5,5), 0)
    variants.append(blur)

    return variants

total_images = 0
used = 0
failed = 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = [f"f{i}" for i in range(1,127)] + ["label"]
    writer.writerow(header)

    for letter in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, letter)
        if not os.path.isdir(folder):
            continue

        print(f"\nProcessing {letter}")

        for img_name in os.listdir(folder):
            total_images += 1
            path = os.path.join(folder, img_name)

            img = cv2.imread(path)
            if img is None:
                failed += 1
                continue

            img = cv2.resize(img, (640, 480))
            detected = None

            for variant in preprocess_variants(img):
                rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect(mp_image)
                if result.hand_landmarks:
                    detected = result
                    break

            if detected is None:
                failed += 1
                continue

            features = extract_126_features(detected)
            writer.writerow(features.tolist() + [letter])
            used += 1

            if used % 500 == 0:
                print("Used samples:", used)

print("\n✅ EXTRACTION COMPLETE")
print("Total images :", total_images)
print("Used samples :", used)
print("Failed images:", failed)
print("Saved CSV    :", OUTPUT_CSV)
