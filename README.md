# BL-37: Multi-Modal Border Intrusion Detection System

A real-time surveillance framework for detecting intrusions in extreme border environments using **camera-based object detection (YOLO)** and **machine learning classification**.

---

## 🚀 Overview

Border surveillance in harsh and remote environments suffers from:

- Limited visibility and accessibility  
- High false alarms (animals, weather, noise)  
- Lack of integration between sensing systems  

This project provides a **Stage 0 prototype** that focuses on:

- Real-time **camera feed processing**
- Object detection using **YOLOv8**
- Feature extraction from detected objects
- Threat classification using a trained ML model
- Live dashboard visualization

---

## 🧠 System Pipeline

1. Camera Feed Capture (OpenCV)
2. Object Detection (YOLOv8)
3. Feature Extraction (Bounding box + heuristics)
4. Feature Scaling (StandardScaler)
5. Classification (Random Forest / ML model)
6. Threat Label Output (Human / Vehicle / Animal / Environmental / Clear)
7. UI Dashboard Display (HTML + JS)

---

## 🏗️ Project Structure

Stage 0/
│
├── stage0_camera.py # Main backend (YOLO + ML pipeline)
├── model.pkl # Trained classifier
├── scaler.pkl # Feature scaler
├── label_encoder.pkl # Label encoder
├── app.py (if Flask used) # API endpoints
│
├── templates/
│ └── index.html # Frontend dashboard
│
└── static/
└── (CSS/JS assets)
---

## ⚙️ Tech Stack

- **Python**
- **OpenCV**
- **Ultralytics YOLOv8**
- **Scikit-learn**
- **Flask (for API)**
- **HTML + CSS + JavaScript**

---

## 🧠 Architecture Diagram
<img width="1022" height="601" alt="image" src="https://github.com/user-attachments/assets/1a98102f-d45b-42cc-b3f6-ece8e5ef17b5" />

---
## 📡 API Endpoints

| Endpoint        | Method | Description                  |
|----------------|--------|------------------------------|
| `/video_feed`  | GET    | Live camera stream           |
| `/predict`     | POST   | Returns latest prediction    |
| `/log`         | GET    | Detection history            |
| `/status`      | GET    | Current detection            |

## 📦 Installation

```bash
pip install opencv-python ultralytics scikit-learn flask numpy```


