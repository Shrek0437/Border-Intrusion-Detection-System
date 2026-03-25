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
