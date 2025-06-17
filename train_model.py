# train_model.py

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

mp_face_detection = mp.solutions.face_detection

label_mapping = {"awake": 0, "drowsy": 1, "sleepy": 2}
input_dataset = 'dataset'

def detect_and_crop_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, box_w, box_h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            return img[max(0, y):y+box_h, max(0, x):x+box_w]
    return None

def resize_and_normalize_image(image, target_size=(224, 224)):
    gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, target_size)
    return resized_face / 255.0

def train_model():
    # Chargement des données
    X, y = [], []
    for class_folder, label in label_mapping.items():
        class_path = os.path.join(input_dataset, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('jpg', 'jpeg', 'png')):
                    face = detect_and_crop_face(os.path.join(class_path, image_file))
                    if face is not None:
                        X.append(resize_and_normalize_image(face))
                        y.append(label)

    X = np.array(X).reshape(-1, 224, 224, 1)
    y = to_categorical(np.array(y), num_classes=3)

    # Création du modèle CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

    model.save("mon_model.h5")
    print("✅ Modèle entraîné et sauvegardé.")

if __name__ == "__main__":
    train_model()
