"""
Stage 0 — Physical Border Intrusion Classifier
BL-37 | Multi-Modal Surveillance Framework for Extreme Border Environments

Trains a Random Forest on perimeter camera object detection features.
Classifies detections into: Human, Vehicle, Animal, Environmental, Clear

Features are derived from what a perimeter camera + object detector would produce:
bounding box geometry, motion vectors, aspect ratio, thermal proxy, time context, zone info, etc.
"""

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SEED = 42
np.random.seed(SEED)

CLASSES = ["Clear", "Human", "Vehicle", "Animal", "Environmental"]
N_SAMPLES = 5000
CLASS_DIST = [0.30, 0.22, 0.15, 0.20, 0.13]

FEATURES = [
    "bbox_width", "bbox_height", "bbox_area", "aspect_ratio",
    "bbox_x_center", "bbox_y_center",
    "motion_magnitude", "motion_direction_x", "motion_direction_y",
    "motion_consistency", "frames_in_motion",
    "edge_density", "vertical_symmetry", "elongation",
    "time_of_day_sin", "time_of_day_cos", "night_mode", "persistence_frames",
    "zone_id", "proximity_to_fence", "crossing_vector",
    "speed_estimate", "path_linearity", "size_change_rate",
    "ambient_brightness", "background_variance", "rain_score", "prev_alert_zone",
]

def make_dataset(n=N_SAMPLES):
    counts = [int(n * p) for p in CLASS_DIST]
    counts[-1] = n - sum(counts[:-1])
    rows, labels = [], []

    def r(lo, hi): return np.random.uniform(lo, hi)
    def ri(lo, hi): return float(np.random.randint(lo, hi+1))
    def rn(mu, sig): return max(0.0, float(np.random.normal(mu, sig)))

    def sample(cls):
        hour = np.random.randint(0, 24)
        night = 1.0 if (hour < 6 or hour > 20) else 0.0
        tod_sin = float(np.sin(2 * np.pi * hour / 24))
        tod_cos = float(np.cos(2 * np.pi * hour / 24))

        if cls == "Human":
            w = r(0.04, 0.12); h = r(0.15, 0.35)
            return [w, h, w*h, w/(h+1e-6), r(0.1,0.9), r(0.3,0.9),
                    rn(0.55,0.2), r(-0.3,0.3), r(-0.1,0.1), r(0.6,1.0), ri(5,40),
                    r(0.4,0.7), r(0.6,1.0), r(0.3,0.7),
                    tod_sin, tod_cos, night, ri(10,80),
                    ri(0,7), r(0.0,0.3), float(np.random.choice([0,1],p=[0.4,0.6])),
                    rn(1.2,0.5), r(0.5,1.0), r(0.0,0.15),
                    r(30,220), r(5,40), r(0.0,0.2), float(np.random.randint(0,2))]

        elif cls == "Vehicle":
            w = r(0.12, 0.45); h = r(0.08, 0.22)
            return [w, h, w*h, w/(h+1e-6), r(0.05,0.95), r(0.4,0.9),
                    rn(1.8,0.6), r(-0.8,0.8), r(-0.1,0.1), r(0.7,1.0), ri(3,25),
                    r(0.5,0.8), r(0.4,0.8), r(1.2,3.5),
                    tod_sin, tod_cos, night, ri(5,50),
                    ri(0,7), r(0.0,0.4), float(np.random.choice([0,1],p=[0.3,0.7])),
                    rn(6.5,2.0), r(0.7,1.0), r(0.05,0.35),
                    r(40,240), r(3,25), r(0.0,0.15), float(np.random.randint(0,2))]

        elif cls == "Animal":
            w = r(0.02, 0.14); h = r(0.02, 0.14)
            return [w, h, w*h, w/(h+1e-6), r(0.05,0.95), r(0.2,0.95),
                    rn(0.7,0.4), r(-0.6,0.6), r(-0.4,0.4), r(0.2,0.7), ri(1,20),
                    r(0.3,0.6), r(0.2,0.6), r(0.4,1.6),
                    tod_sin, tod_cos, night, ri(1,30),
                    ri(0,7), r(0.0,0.8), float(np.random.choice([0,1],p=[0.85,0.15])),
                    rn(1.8,1.0), r(0.1,0.6), r(-0.05,0.1),
                    r(20,240), r(8,50), r(0.0,0.15), 0.0]

        elif cls == "Environmental":
            w = r(0.05, 0.5); h = r(0.05, 0.4)
            return [w, h, w*h, w/(h+1e-6), r(0.0,1.0), r(0.0,1.0),
                    rn(0.3,0.25), r(-0.5,0.5), r(-0.5,0.5), r(0.0,0.35), ri(1,8),
                    r(0.1,0.4), r(0.1,0.5), r(0.5,2.0),
                    tod_sin, tod_cos, night, ri(1,10),
                    ri(0,7), r(0.2,1.0), 0.0,
                    rn(0.2,0.15), r(0.0,0.4), r(-0.1,0.1),
                    r(10,255), r(30,120), r(0.3,1.0), 0.0]

        else:  # Clear
            return [0.0,0.0,0.0,1.0, r(0,1), r(0,1),
                    r(0.0,0.08), r(-0.05,0.05), r(-0.05,0.05), 0.0, 0.0,
                    r(0.0,0.15), r(0.0,0.3), 1.0,
                    tod_sin, tod_cos, night, 0.0,
                    ri(0,7), r(0.3,1.0), 0.0,
                    0.0, 0.0, 0.0,
                    r(80,200), r(2,15), r(0.0,0.3), 0.0]

    for cls, cnt in zip(CLASSES, counts):
        for _ in range(cnt):
            rows.append(sample(cls))
            labels.append(cls)

    df = pd.DataFrame(rows, columns=FEATURES)
    df["label"] = labels
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("Generating perimeter camera detection dataset...")
df = make_dataset()
print(f"  Samples : {len(df)}")
print(f"  Classes :\n{df['label'].value_counts().to_string()}\n")

X = df[FEATURES].values
y = df["label"].values
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Training Random Forest...")
clf = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    min_samples_split=4, class_weight="balanced",
    random_state=SEED, n_jobs=-1)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

fi = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(fi.head(10).round(4).to_string())

for obj, name in [(clf,"model"),(scaler,"scaler"),(le,"label_encoder")]:
    with open(f"{name}.pkl","wb") as f: pickle.dump(obj,f)

meta = {
    "features": FEATURES,
    "classes": list(le.classes_),
    "accuracy": round(acc*100, 2),
    "feature_importances": fi.head(14).round(4).to_dict(),
    "classification_report": classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True),
    "confusion_matrix": cm.tolist(),
}
with open("model_meta.json","w") as f: json.dump(meta,f,indent=2)
print("\nSaved: model.pkl, scaler.pkl, label_encoder.pkl, model_meta.json")
