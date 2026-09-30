# AutoVision-AI

**Real-Time Autonomous Vehicle Perception on Embedded Hardware**

AutoVision-AI is an AI-powered computer vision system designed for real-time autonomous vehicle perception on resource-constrained hardware such as the **Raspberry Pi 4**.

The system uses a custom-trained **YOLOv8 object detection model** to detect **traffic lights** and **stop signs**. For detected traffic lights, **OpenCV-based HSV color analysis** is used to determine whether the signal is **Red, Yellow, or Green**.

The project focuses on deploying modern computer vision models efficiently on CPU-only embedded hardware without relying on GPUs or external accelerators.

---

## Overview

Autonomous vehicles require fast and reliable perception to understand their surroundings. However, deploying deep learning models on embedded platforms introduces significant computational constraints.

AutoVision-AI addresses this challenge by combining:

- YOLOv8 for object detection
- HSV-based image processing for traffic light state classification
- Frame skipping for reduced inference overhead
- Bounding-box filtering to reduce false detections
- Low-resolution inference for improved CPU performance
- Bounding-box reuse between inference frames
- Real-time visualization and decision output

On a Raspberry Pi 4 CPU, these optimizations allow the system to achieve approximately **10–12 YOLO inference FPS** without hardware acceleration.

---

## Features

### Traffic Light Detection

A custom-trained YOLOv8 model detects traffic lights from the camera feed in real time.

### Stop Sign Detection

The same object detection model identifies stop signs and provides their locations using bounding boxes.

### Traffic Light State Classification

After detecting a traffic light, the corresponding image region is processed using HSV color segmentation to classify its state:

- Red
- Yellow
- Green

### Real-Time Decision Output

The detected traffic objects and their corresponding states are processed by decision logic to determine the action that the autonomous vehicle should take.

### Embedded Deployment

The system is designed to operate on the Raspberry Pi 4 using CPU-only inference, making it suitable for low-cost embedded robotic platforms.

### Real-Time Visualization

Detection results, traffic light states, and vehicle actions are displayed in real time.

---

## System Architecture

```text
                    Camera Input
                         |
                         v
                +------------------+
                |     YOLOv8       |
                | Object Detection |
                +------------------+
                         |
                         v
              +-----------------------+
              | Traffic Light / Stop  |
              |      Sign Detection   |
              +-----------------------+
                    |           |
                    |           |
            Traffic Light    Stop Sign
                    |
                    v
             +-------------+
             | HSV Color   |
             |  Analysis   |
             +-------------+
                    |
                    v
            Red / Yellow / Green
                    |
                    v
             +-------------+
             |   Decision  |
             |    Logic    |
             +-------------+
                    |
                    v
        Display Output / Vehicle Control
```

---

## Performance Optimizations

Running deep learning inference on a Raspberry Pi requires careful optimization. AutoVision-AI implements several techniques to reduce computational overhead.

### 1. Frame Skipping

YOLO inference is computationally expensive. Instead of performing detection on every camera frame, inference is performed periodically while intermediate frames reuse the most recent detection results.

This reduces the overall inference workload while maintaining responsive visualization.

### 2. Bounding-Box Filtering

Extremely large or invalid detections can produce false positives. Bounding-box size filtering is therefore applied to reject detections that do not satisfy predefined constraints.

### 3. HSV-Based Color Analysis

Rather than using an additional neural network to classify the state of a traffic light, the detected traffic light region is analyzed using the HSV color space.

HSV provides a convenient representation for separating color information from brightness, allowing the system to identify dominant red, yellow, and green regions.

### 4. Low-Resolution Inference

The camera pipeline uses a resolution of **192 × 144** for YOLO inference. Processing fewer pixels significantly reduces CPU computation.

A separate **640 × 480** stream is used for OpenCV visualization.

### 5. Bounding-Box Reuse

Previously detected bounding boxes are reused between YOLO inference frames. This avoids unnecessary detection computation while keeping the displayed results responsive.

---

## Hardware

### Current Hardware

| Component | Description |
|---|---|
| Processing Unit | Raspberry Pi 4 (2GB) |
| Camera | Raspberry Pi Camera Module |
| Display | 2.4" LCD Display |
| Storage | MicroSD Card |
| Power | Raspberry Pi-compatible power supply |
| Connectivity | Jumper wires |

### Planned Hardware

The next stage of the platform will integrate:

- Motor driver module
- DC motors
- Robot chassis
- Ultrasonic sensors
- Additional vehicle control hardware

These components will allow the perception system to be connected to a physical autonomous vehicle platform.

---

## Software Stack

| Technology | Purpose |
|---|---|
| Python | Application and control logic |
| YOLOv8 | Object detection |
| Ultralytics | YOLO model training and inference |
| OpenCV | Image processing and visualization |
| PyTorch | Deep learning framework |
| Raspberry Pi OS | Embedded operating environment |

---

## Model Training

The detection model was trained using custom datasets containing images of:

- Traffic lights
- Stop signs

Transfer learning was used with a pretrained YOLOv8 model as the starting point.

### Training Configuration

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=30 imgsz=320 batch=8
```

The lightweight **YOLOv8n** architecture was selected to balance detection performance with the computational limitations of Raspberry Pi hardware.

---

## Performance

| Metric | Result |
|---|---|
| Camera Frame Rate | ~30 FPS |
| YOLO Inference Rate | ~10–12 FPS |
| Inference Resolution | 192 × 144 |
| Display Resolution | 640 × 480 |
| Hardware Acceleration | Not used |
| Processing Hardware | Raspberry Pi 4 CPU |

> **Note:** Actual performance can vary depending on lighting conditions, model configuration, background complexity, and Raspberry Pi system load.

---

## Project Structure

```text
AutoVision-AI/
│
├── models/
│   └── autovision_model.pt
│
├── scripts/
│   ├── webcam_test.py
│   ├── autovision_model.pt
│   └── lcd_display.py
│
├── datasets/
│   └── autovision_dataset/
│
├── images/
│   └── demo_images/
│
├── README.md
└── requirements.txt
```

---

## Detection and Decision Flow

The system follows the following processing pipeline:

```text
Camera Frame
     |
     v
Preprocessing
     |
     v
YOLOv8 Detection
     |
     +----------------------+
     |                      |
     v                      v
Traffic Light           Stop Sign
     |                      |
     v                      v
HSV Analysis           Stop Decision
     |
     v
Red / Yellow / Green
     |
     v
Vehicle Decision
     |
     v
Display / Vehicle Control
```

---

## Example Behavior

The perception system can identify traffic-related objects and use the detected information for basic autonomous driving decisions.

| Detection | System Response |
|---|---|
| Red Traffic Light | Stop |
| Yellow Traffic Light | Prepare to stop / slow down |
| Green Traffic Light | Proceed |
| Stop Sign | Stop |

The decision layer can later be extended with additional vehicle state, distance, speed, and obstacle information.

---

## Future Development

The current system focuses primarily on visual perception. Future development will extend the platform toward a more complete autonomous vehicle system.

### Vehicle Control

Integrate the perception system with motor drivers and DC motors to enable physical vehicle control.

### Obstacle Detection

Integrate ultrasonic sensors to detect obstacles and provide an additional safety layer for autonomous navigation.

### Decision-Based Driving

Develop a centralized decision-making system that combines traffic signals, stop signs, obstacle information, and vehicle state.

### Lane Detection

Add lane detection and tracking to support road-following behavior.

### Navigation

Extend the system with path planning and navigation capabilities.

### Telemetry

Add remote monitoring for vehicle status, detections, performance metrics, and system diagnostics.

---

## Applications

The techniques demonstrated by AutoVision-AI can be applied to:

- Educational autonomous vehicle platforms
- Embedded computer vision systems
- Low-cost robotics
- Traffic sign and signal recognition
- Edge AI experimentation
- Autonomous navigation research

---

## Limitations

Because the system is designed for low-power embedded hardware, there are several practical limitations:

- CPU-only inference limits the achievable detection rate.
- Low-resolution inference can reduce detection accuracy for distant objects.
- HSV-based traffic light classification can be affected by lighting conditions, reflections, and environmental colors.
- The current decision logic is intended as a prototype and is not suitable for deployment in real-world road traffic.
- Additional sensors and safety mechanisms are required for reliable autonomous vehicle operation.

---

## Getting Started

### Clone the Repository

```bash
git clone <repository-url>
cd AutoVision-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Detection System

```bash
python scripts/webcam_test.py
```

Make sure the Raspberry Pi camera is properly connected and enabled before starting the system.

---

## License

This project is open-source and distributed under the **MIT License**.

---

## Author

Developed as part of an **AI and Computer Vision autonomous driving project**, with a focus on deploying deep learning-based perception systems efficiently on embedded robotic platforms.

