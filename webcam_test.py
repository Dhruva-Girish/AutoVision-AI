from ultralytics import YOLO
import cv2
import numpy as np
from picamera2 import Picamera2
from lcd_alerts import stop_sign, red_light, yellow_light, green_light, clear
import time

print("AutoVision AI started")

model = YOLO("autovision_model.pt")

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640,480)},
    lores={"format": "YUV420", "size": (192,144)}
)

picam2.configure(config)
picam2.start()

cv2.namedWindow("Traffic Sign Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Sign Detector",960,720)

frame_count = 0

last_boxes = []
last_labels = []

current_display = "NONE"

lcd_busy = False
lcd_finish_time = 0


while True:

    frame = picam2.capture_array("main")

    ai_frame = picam2.capture_array("lores")
    ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_YUV2BGR_I420)

    frame_count += 1

    detected_state = "NONE"

    # ---------------- YOLO DETECTION ----------------

    if frame_count % 5 == 0 and not lcd_busy:

        results = model(ai_frame, imgsz=192, conf=0.55, verbose=False)[0]

        last_boxes = []
        last_labels = []

        scale_x = frame.shape[1] / ai_frame.shape[1]
        scale_y = frame.shape[0] / ai_frame.shape[0]

        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "stop_sign":

                detected_state = "STOP"
                last_boxes.append((x1,y1,x2,y2))
                last_labels.append("STOP")
                break

            pad = 10

            x1p = max(0, x1-pad)
            y1p = max(0, y1-pad)
            x2p = min(frame.shape[1], x2+pad)
            y2p = min(frame.shape[0], y2+pad)

            roi = frame[y1p:y2p, x1p:x2p]

            if roi.size == 0:
                continue

            roi = cv2.GaussianBlur(roi,(5,5),0)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            red1 = cv2.inRange(hsv,(0,150,120),(10,255,255))
            red2 = cv2.inRange(hsv,(160,150,120),(180,255,255))
            red_mask = red1 + red2

            yellow_mask = cv2.inRange(hsv,(18,120,120),(35,255,255))
            green_mask = cv2.inRange(hsv,(40,80,80),(90,255,255))

            red_pixels = cv2.countNonZero(red_mask)
            yellow_pixels = cv2.countNonZero(yellow_mask)
            green_pixels = cv2.countNonZero(green_mask)

            m = max(red_pixels,yellow_pixels,green_pixels)

            if m == 0:
                continue

            if m == red_pixels:
                detected_state = "RED"

            elif m == yellow_pixels:
                detected_state = "YELLOW"

            else:
                detected_state = "GREEN"

            last_boxes.append((x1,y1,x2,y2))
            last_labels.append(detected_state)

            break


    # ---------------- LCD CONTROL ----------------

    if detected_state != current_display and not lcd_busy:

        lcd_busy = True

        if detected_state == "STOP":
            stop_sign()
            lcd_finish_time = time.time() + 3

        elif detected_state == "RED":
            red_light()
            lcd_finish_time = time.time() + 3

        elif detected_state == "YELLOW":
            yellow_light()
            lcd_finish_time = time.time() + 3

        elif detected_state == "GREEN":
            green_light()
            lcd_finish_time = time.time() + 3

        current_display = detected_state


    # unlock LCD after message finishes

    if lcd_busy and time.time() > lcd_finish_time:

        lcd_busy = False

        if current_display != "NONE":
            clear()
            current_display = "NONE"


    # ---------------- DRAW BOXES ----------------

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
