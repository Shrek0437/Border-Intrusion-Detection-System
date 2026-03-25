from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO
from collections import deque
from datetime import datetime

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

# Global state
latest_result = {
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

# Detection log
detection_log = deque(maxlen=20)
last_label = None


def map_label(yolo_label):
    if yolo_label == "person":
        return "Human"
    elif yolo_label in ["car", "bus", "truck", "motorbike"]:
        return "Vehicle"
    elif yolo_label in ["dog", "cat", "bird"]:
        return "Animal"
    else:
        return "Environmental"


def generate_frames():
    global latest_result, last_label

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)[0]

        best_label = "Clear"
        max_conf = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf > max_conf:
                max_conf = conf
                best_label = label

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        final_label = map_label(best_label)

        confidence = round(max_conf * 100, 2)

        # Update latest result
        latest_result = {
            "prediction": final_label,
            "confidence": confidence,
            "probabilities": {
                "Clear": 1 if final_label == "Clear" else 0,
                "Human": 1 if final_label == "Human" else 0,
                "Vehicle": 1 if final_label == "Vehicle" else 0,
                "Animal": 1 if final_label == "Animal" else 0,
                "Environmental": 1 if final_label == "Environmental" else 0
            }
        }

        # Add to log only if changed
        if final_label != last_label:
            detection_log.appendleft({
                "label": final_label,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            last_label = final_label

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(latest_result)


@app.route('/log')
def get_log():
    return jsonify(list(detection_log))


if __name__ == '__main__':
    app.run(debug=True)