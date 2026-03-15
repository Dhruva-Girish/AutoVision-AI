# AutoVision-AI 🚗🤖

**AutoVision-AI** is an AI-powered computer vision system designed for autonomous vehicle perception.
The project detects **traffic lights** and **stop signs** in real time using a camera feed and makes driving decisions based on the detected signals.

The system is built using **YOLOv8**, **OpenCV**, and **Raspberry Pi**, enabling lightweight real-time inference suitable for robotic and embedded autonomous platforms.

---

## Features

* 🚦 **Traffic Light Detection** using YOLOv8 object detection
* 🛑 **Stop Sign Detection** using a custom-trained model
* 🎥 **Real-time camera inference**
* 🎨 **Traffic light color recognition** using OpenCV
* 📺 **Interactive display output** for vehicle decisions
* 🤖 Designed for **Raspberry Pi-based autonomous vehicles**

---

## System Architecture

Camera Input
->
YOLO Object Detection
->
Traffic Light / Stop Sign Detection
->
OpenCV Color Recognition (for traffic lights)
->
Decision Logic
->
Display Output / Vehicle Control

---

## Technology Stack

* **Python**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **PyTorch**
* **Raspberry Pi 4 2BG**
* **SPI TFT Display**
* **Raspberry Pi Camera Module**

---

## Hardware Components

* Raspberry Pi 4
* Raspberry Pi Camera Module
* 2.4" SPI TFT Display
* Jumper wires
* MicroSD card
* Power supply

Future additions:

* Motor driver module
* DC motors and chassis
* Ultrasonic sensors for obstacle detection

---

## Model Training

The detection model was trained using **custom datasets** containing images of:

* Traffic lights
* Stop signs

The model uses **transfer learning**, starting from a pretrained YOLOv8 network and fine-tuning it with the combined dataset.

Training command:

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=30 imgsz=320 batch=8
```

---

## Example Detection

The system can detect:

* 🚦 Traffic Lights
* 🛑 Stop Signs

Traffic light colors are further analyzed using **OpenCV image processing** to determine:

* Red
* Yellow
* Green

---

## Project Structure

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

## Author

Developed as part of an **AI and Computer Vision autonomous driving project**.

---

## License

This project is open-source and available under the **MIT License**.
