from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("autovision_model.pt")

cap = cv2.VideoCapture(0)

last_boxes = []
last_colors = []

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,480))

    results = model(frame)[0]

    last_boxes = []
    last_colors = []

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])
        label = model.names[cls]

        # -------------------------
        # STOP SIGN DETECTION
        # -------------------------
        if label == "stop_sign":
            last_boxes.append((x1,y1,x2,y2))
            last_colors.append("STOP")
            continue

        # -------------------------
        # TRAFFIC LIGHT DETECTION
        # -------------------------

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Blur removes webcam noise
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

        # Detect all lights
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


    for i in range(len(last_boxes)):

        x1,y1,x2,y2 = last_boxes[i]
        color = last_colors[i]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.putText(frame,color,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)


    cv2.imshow("Traffic Sign Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()