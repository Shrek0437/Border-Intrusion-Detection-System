"""
Stage 0 — Border Intrusion Prediction Engine
Usage: python3 stage0_predict.py '{"bbox_width":0.07,"bbox_height":0.25,...}'
Returns JSON: predicted class, confidence, per-class probabilities.
"""
import sys, json, pickle
import numpy as np

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

def predict(d):
    with open("model.pkl","rb") as f: clf=pickle.load(f)
    with open("scaler.pkl","rb") as f: scaler=pickle.load(f)
    with open("label_encoder.pkl","rb") as f: le=pickle.load(f)
    vec = np.array([float(d.get(k,0)) for k in FEATURES]).reshape(1,-1)
    vec_s = scaler.transform(vec)
    idx = clf.predict(vec_s)[0]
    probs = clf.predict_proba(vec_s)[0]
    label = le.inverse_transform([idx])[0]
    return {
        "prediction": label,
        "confidence": round(float(probs[idx])*100, 2),
        "probabilities": {c: round(float(p)*100,2) for c,p in zip(le.classes_, probs)}
    }

if __name__ == "__main__":
    try:
        data = json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
        print(json.dumps(predict(data)))
    except Exception as e:
        print(json.dumps({"error": str(e)})); sys.exit(1)
