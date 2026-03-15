from ultralytics import YOLO
import cv2
import numpy as np
from picamera2 import Picamera2

print("Program started")

# Load model
model = YOLO("autovision_model.pt")

# Initialize Pi Camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (320, 240)}
)

picam2.configure(config)
picam2.start()

# Create larger window
cv2.namedWindow("Traffic Sign Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Sign Detector", 960, 720)

last_boxes = []
last_colors = []

frame_count = 0

while True:

    frame = picam2.capture_array()
    frame_count += 1

    # Run YOLO every 3 frames for speed
    if frame_count % 3 == 0:

        results = model(frame, imgsz=320, conf=0.4)[0]

        last_boxes = []
        last_colors = []

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            label = model.names[cls]

            # STOP SIGN DETECTION
            if label == "stop_sign":
                last_boxes.append((x1,y1,x2,y2))
                last_colors.append("STOP")
                continue

            # TRAFFIC LIGHT DETECTION
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            roi = cv2.GaussianBlur(roi,(5,5),0)

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape

            top = gray[0:int(h/3), :]
            middle = gray[int(h/3):int(2*h/3), :]
            bottom = gray[int(2*h/3):h, :]

            top_b = np.mean(top)
            mid_b = np.mean(middle)
            bot_b = np.mean(bottom)

            bright_threshold = 200

            if top_b > bright_threshold and mid_b > bright_threshold and bot_b > bright_threshold:
                color = "RYG"

            else:

                max_val = max(top_b, mid_b, bot_b)

                if max_val == top_b:
                    color = "RED"

                elif max_val == mid_b:
                    color = "YELLOW"

                else:
                    color = "GREEN"

            last_boxes.append((x1,y1,x2,y2))
            last_colors.append(color)

    # Draw boxes
    for i in range(len(last_boxes)):

        x1,y1,x2,y2 = last_boxes[i]
        color = last_colors[i]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.putText(frame,color,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,255,0),2)

    # Resize only for display
    display = cv2.resize(frame,(960,720))

    cv2.imshow("Traffic Sign Detector",display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
