from ultralytics import YOLO
import cv2
import numpy as np
from picamera2 import Picamera2

print("AutoVision AI started")

model = YOLO("autovision_model.pt")

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (320,240)}
)

picam2.configure(config)
picam2.start()

cv2.namedWindow("Traffic Sign Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Sign Detector",960,720)

frame_count = 0

last_boxes = []
last_labels = []

while True:

    frame = picam2.capture_array()
    frame_count += 1

    # Run YOLO every 3 frames
    if frame_count % 3 == 0:

        results = model(frame, imgsz=320, conf=0.6, verbose=False)[0]

        last_boxes = []
        last_labels = []

        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            # Remove giant false detections
            box_area = (x2-x1)*(y2-y1)
            frame_area = frame.shape[0]*frame.shape[1]

            if box_area > 0.6*frame_area:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "stop_sign":

                last_boxes.append((x1,y1,x2,y2))
                last_labels.append("STOP")
                continue

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            roi = cv2.GaussianBlur(roi,(5,5),0)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            red1 = cv2.inRange(hsv,(0,100,100),(10,255,255))
            red2 = cv2.inRange(hsv,(160,100,100),(180,255,255))
            red_mask = red1 + red2

            yellow_mask = cv2.inRange(hsv,(20,100,100),(35,255,255))
            green_mask = cv2.inRange(hsv,(40,70,70),(90,255,255))

            red_pixels = cv2.countNonZero(red_mask)
            yellow_pixels = cv2.countNonZero(yellow_mask)
            green_pixels = cv2.countNonZero(green_mask)

            m = max(red_pixels,yellow_pixels,green_pixels)

            if m == 0:
                continue

            if m == red_pixels:
                color = "RED"
            elif m == yellow_pixels:
                color = "YELLOW"
            else:
                color = "GREEN"

            last_boxes.append((x1,y1,x2,y2))
            last_labels.append(color)

    # Draw detections (reused between frames)

    for i in range(len(last_boxes)):

        x1,y1,x2,y2 = last_boxes[i]
        label = last_labels[i]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

    cv2.imshow("Traffic Sign Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
