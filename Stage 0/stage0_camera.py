import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from collections import deque
from datetime import datetime

detection_log = deque(maxlen=20)  # keeps last 20 detections

latest_prediction = {
    "prediction": "Clear",
    "confidence": 0,
    "probabilities": {}
}

latest_detection = {
    "prediction": "Clear",
    "confidence": 0,
    "probabilities": {
        "Clear": 1,
        "Human": 0,
        "Vehicle": 0,
        "Animal": 0,
        "Environmental": 0
    }
}

def get_latest_prediction():
    return latest_prediction


# ── Load models ─────────────────────────
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

yolo_model = YOLO("yolov8n.pt")

FEATURES = [
    "bbox_width","bbox_height","bbox_area","aspect_ratio",
    "bbox_x_center","bbox_y_center",
    "motion_magnitude","motion_direction_x","motion_direction_y",
    "motion_consistency","frames_in_motion",
    "edge_density","vertical_symmetry","elongation",
    "time_of_day_sin","time_of_day_cos","night_mode","persistence_frames",
    "zone_id","proximity_to_fence","crossing_vector",
    "speed_estimate","path_linearity","size_change_rate",
    "ambient_brightness","background_variance","rain_score","prev_alert_zone",
]


def extract_features(x1, y1, x2, y2, frame):
    h, w = frame.shape[:2]

    bbox_width = (x2 - x1) / w
    bbox_height = (y2 - y1) / h

    return {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_width * bbox_height,
        "aspect_ratio": bbox_width / (bbox_height + 1e-6),
        "bbox_x_center": ((x1 + x2)/2) / w,
        "bbox_y_center": ((y1 + y2)/2) / h,
        "motion_magnitude": 0.5,
        "motion_direction_x": 0.1,
        "motion_direction_y": 0.1,
        "motion_consistency": 0.7,
        "frames_in_motion": 10,
        "edge_density": 0.5,
        "vertical_symmetry": 0.7,
        "elongation": 0.5,
        "time_of_day_sin": 0.5,
        "time_of_day_cos": 0.8,
        "night_mode": 0,
        "persistence_frames": 10,
        "zone_id": 2,
        "proximity_to_fence": 0.2,
        "crossing_vector": 1,
        "speed_estimate": 1.0,
        "path_linearity": 0.7,
        "size_change_rate": 0.05,
        "ambient_brightness": 120,
        "background_variance": 10,
        "rain_score": 0.1,
        "prev_alert_zone": 0,
    }


def predict(features):
    vec = np.array([features[k] for k in FEATURES]).reshape(1, -1)
    vec = scaler.transform(vec)

    idx = clf.predict(vec)[0]
    probs = clf.predict_proba(vec)[0]

    label = le.inverse_transform([idx])[0]
    confidence = float(probs[idx]) * 100

    return label, confidence, {
        c: float(p)*100 for c,p in zip(le.classes_, probs)
    }


def main():
    global latest_prediction

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 320))

        # 🔥 Run YOLO every 5 frames
        if frame_count % 5 == 0:
            results = yolo_model(frame, verbose=False)

            label = "Clear"
            confidence = 0

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                detected_class = yolo_model.names[cls_id]

                # Map YOLO → your labels
                if detected_class == "person":
                    yolo_label = "Human"
                elif detected_class in ["car", "truck", "bus", "motorbike"]:
                    yolo_label = "Vehicle"
                elif detected_class in ["dog", "cat", "cow", "horse"]:
                    yolo_label = "Animal"
                else:
                    yolo_label = "Environmental"

                # 🔥 THIS IS WHAT YOU WERE MISSING
                features = extract_features(x1, y1, x2, y2, frame)

                pred_label, pred_conf, probs = predict(features)

                # update global prediction
                latest_prediction = {
                    "prediction": pred_label,
                    "confidence": pred_conf,
                    "probabilities": probs
                }

                # logging (use ML output, not YOLO guess)
                detection_log.appendleft({
                    "label": pred_label,
                    "confidence": round(pred_conf, 2),
                    "time": datetime.now().strftime("%H:%M:%S")
                })

                detected_class = yolo_model.names[cls_id]

                if detected_class == "person":
                    label = "Human"
                elif detected_class in ["car", "truck", "bus", "motorbike"]:
                    label = "Vehicle"
                elif detected_class in ["dog", "cat", "cow", "horse"]:
                    label = "Animal"
                else:
                    label = "Environmental"

                confidence = round(conf * 100, 2)
            
            detection_log.appendleft({"label": label,"confidence": confidence, "time": datetime.now().strftime("%H:%M:%S")})

        frame_count += 1