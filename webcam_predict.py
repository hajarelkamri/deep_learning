# webcam_predict.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("mon_model.h5")
mp_face_detection = mp.solutions.face_detection

label_mapping = {"awake": 0, "drowsy": 1, "sleepy": 2}
label_mapping_reverse = {v: k for k, v in label_mapping.items()}

def detect_and_crop_face(img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, box_w, box_h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            cropped_face = img[max(0, y):y+box_h, max(0, x):x+box_w]
            return cropped_face, (x, y, box_w, box_h)
    return None, None

def resize_and_normalize_image(image, target_size=(224, 224)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    return resized / 255.0

cap = cv2.VideoCapture(0)
print("📷 Webcam ouverte. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, bbox = detect_and_crop_face(frame)
    if face is not None and bbox is not None:
        x, y, box_w, box_h = bbox
        input_img = resize_and_normalize_image(face).reshape(1, 224, 224, 1)

        predictions = model.predict(input_img)
        predicted_label = np.argmax(predictions[0])
        predicted_class = label_mapping_reverse[predicted_label]

        color_map = {"awake": (0, 255, 0), "drowsy": (0, 165, 255), "sleepy": (0, 0, 255)}
        color = color_map.get(predicted_class, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
        cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Webcam - Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
