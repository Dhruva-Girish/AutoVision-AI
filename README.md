# AutoVision-AI 🚗🤖

**AutoVision-AI** is an optimized AI-powered computer vision system designed for **real-time autonomous vehicle perception on low-power hardware** such as the **Raspberry Pi 4**.

The system detects **traffic lights** and **stop signs** using a custom-trained **YOLOv8 model** and determines the **traffic light state (Red / Yellow / Green)** using **OpenCV HSV color analysis**.
It is optimized to run **efficiently on CPU-only systems** without requiring GPUs or external accelerators.

The project demonstrates how modern computer vision models can be deployed on **embedded robotics platforms** for real-time decision-making.

---

# 🚀 Key Optimizations

To make the system run smoothly on Raspberry Pi hardware, several performance optimizations were implemented:

* ⚡ **Frame skipping (50%)** to increase inference speed
* ⚡ **False detection filtering** to remove oversized bounding boxes
* ⚡ **HSV-based color detection** for accurate traffic light classification
* ⚡ **Low-resolution camera pipeline (320×240)** for faster inference
* ⚡ **Efficient bounding-box reuse between frames**

These improvements allow the system to reach approximately:

**12–16 YOLO inference FPS on Raspberry Pi 4 CPU**

without hardware acceleration.

---

# Features

### 🚦 Traffic Light Detection

Uses a custom-trained **YOLOv8 model** to detect traffic lights in real time.

### 🛑 Stop Sign Detection

Detects stop signs using the same trained model.

### 🎨 Traffic Light Color Classification

OpenCV analyzes the detected traffic light region using **HSV color segmentation** to determine:

* 🔴 Red
* 🟡 Yellow
* 🟢 Green

### 📊 Real YOLO FPS Monitoring

Displays the **actual YOLO inference speed** on the screen.

### ⚡ Optimized for Embedded Systems

Designed specifically for **Raspberry Pi CPU-only environments**.

### 📺 Real-Time Visualization

Bounding boxes and traffic decisions are displayed on a live camera feed.

---

# System Architecture

Camera Input
↓
YOLOv8 Object Detection
↓
Traffic Light / Stop Sign Classification
↓
HSV Color Detection (OpenCV)
↓
Decision Logic
↓
Display Output / Vehicle Control

---

# Technology Stack

**Languages & Frameworks**

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* PyTorch

**Hardware Platform**

* Raspberry Pi 4 (2GB)
* Raspberry Pi Camera Module
* SPI TFT Display

---

# Hardware Components

### Current Hardware

* Raspberry Pi 4
* Raspberry Pi Camera Module
* 2.4" SPI TFT Display
* Jumper Wires
* MicroSD Card
* Power Supply

### Planned Hardware

* Motor driver module
* DC motors
* Robot chassis
* Ultrasonic sensors (ADAS obstacle detection)

---

# Model Training

The detection model was trained using **custom datasets** containing images of:

* Traffic lights
* Stop signs

Training was performed using **transfer learning** from a pretrained YOLOv8 model.

Training command:

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=30 imgsz=320 batch=8
```

---

# Performance

| Component          | Performance                   |
| ------------------ | ----------------------------- |
| Camera FPS         | ~30 FPS                       |
| YOLO Inference FPS | ~12–16 FPS                    |
| Frame Skipping     | 1 frame skipped per inference |
| Hardware           | Raspberry Pi 4 CPU            |

---

# Project Structure

```
AutoVision-AI/

models/
   autovision_model.pt

scripts/
   webcam_test.py
   traffic_light_detection.py

datasets/
   autovision_dataset/

images/
   demo_images/

README.md
requirements.txt
```

---

# Example Output

The system can detect:

* 🚦 Traffic lights
* 🛑 Stop signs

And determine traffic light states:

* 🔴 Red
* 🟡 Yellow
* 🟢 Green

All results are displayed in real time with bounding boxes and labels.

---

# Future Development

Upcoming features planned for the autonomous vehicle platform:

* 🚗 Motor control integration
* 🚧 Ultrasonic obstacle detection
* 🧠 Decision-based driving logic
* 📡 Remote monitoring / telemetry
* 🗺️ Lane detection and navigation

---

# Author

Developed as part of an **AI and Computer Vision autonomous driving project** focused on deploying deep learning models on embedded robotic platforms.

---

# License

This project is open-source and distributed under the **MIT License**.
