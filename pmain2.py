from ultralytics import YOLO
import cv2
import cvzone
import pandas as pd

# Your model
model = YOLO("yolo11m.pt")

# Camera RTSP URL
CAM_URL = "rtsp://root:YoloTracking@169.254.41.207/axis-media/media.amp"

# Load COCO classes
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Detection settings
MIN_CONFIDENCE = 0.15
MIN_AREA = 200        # Allow smaller visible parts (closer/partial people)
MAX_AREA = float('inf')  # Remove upper area filter for close-up detection
ASPECT_RATIO_MIN = 0.15  # Allow wider boxes for partial/close people
ASPECT_RATIO_MAX = 10.0  # Allow even taller boxes for close-up

frame_count = 0

# Use YOLO’s built-in streaming
for results in model.predict(CAM_URL, stream=True, conf=0.15, iou=0.45, verbose=False):
    frame_count += 1
    frame = results.orig_img.copy()  # The actual frame from the stream
    a = results.boxes.data
    px = pd.DataFrame(a).astype("float")

    people_count = 0

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        confidence = float(row[4])
        cls_id = int(row[5])
        c = class_list[cls_id]

        if c == "person" and confidence > MIN_CONFIDENCE:
            width = x2 - x1
            height = y2 - y1
            bbox_area = width * height
            aspect_ratio = height / width if width > 0 else 0

            valid_detection = True
            if bbox_area < MIN_AREA:
                valid_detection = False
            if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                valid_detection = False
            if width < 8 or height < 16:  # More permissive for small/partial detections
                valid_detection = False
            # Edge filter: allow detections at the very edge (remove this filter)

            if valid_detection:
                people_count += 1
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                # Confidence-based color
                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.5:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(frame, f'Person {confidence:.2f}', (x1, y1-10), 1, 1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Overlay info
    cvzone.putTextRect(frame, f'Total People: {people_count}', (20, 50), 2, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'Frame: {frame_count}', (20, 100), 1, 1, colorR=(255, 0, 0))

    frame = cv2.resize(frame, (1020, 600))  # or another width/height you prefer

    cv2.imshow("RTSP Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
