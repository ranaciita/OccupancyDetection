import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np

model = YOLO("yolov10s.pt")  

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
       # print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#cap = cv2.VideoCapture(0)  # Use webcam
cap = cv2.VideoCapture('tr1.mp4')  # Use video file

# Check if camera/video file exists
if not cap.isOpened():
    print("Error: Could not open camera or video file")
    exit()

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Settings for detection quality
MIN_CONFIDENCE = 0.3  # Lower threshold to catch more people
MIN_AREA = 800       # Smaller minimum area for distant people
MAX_AREA = 200000    # Maximum area to filter out false detections
ASPECT_RATIO_MIN = 0.3  # Minimum height/width ratio for person shape
ASPECT_RATIO_MAX = 4.0  # Maximum height/width ratio for person shape

# YOLO detection settings
YOLO_CONF = 0.25      # YOLO confidence threshold
YOLO_IOU = 0.45       # Non-maximum suppression threshold

frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    
    # Preprocessing to improve detection
    # Enhance contrast and brightness slightly
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    
    # Optional: Apply slight blur to reduce noise
    # enhanced_frame = cv2.GaussianBlur(enhanced_frame, (3, 3), 0)
    
    # Run YOLO detection with improved settings
    results = model(enhanced_frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    # Count people in current frame
    people_count = 0
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = float(row[4])
        d = int(row[5])
        c = class_list[d]
        
        # Filter for people with good confidence
        if c == 'person' and confidence > MIN_CONFIDENCE:
            # Calculate bounding box dimensions
            width = x2 - x1
            height = y2 - y1
            bbox_area = width * height
            aspect_ratio = height / width if width > 0 else 0
            
            # Multiple filters for better accuracy
            valid_detection = True
            
            # Size filters
            if bbox_area < MIN_AREA or bbox_area > MAX_AREA:
                valid_detection = False
            
            # Shape filter (people are usually taller than wide)
            if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                valid_detection = False
            
            # Minimum dimensions filter
            if width < 20 or height < 40:  # Too small to be a person
                valid_detection = False
            
            # Position filter (avoid detections at very edge of frame)
            if x1 < 5 or y1 < 5 or x2 > (1020-5) or y2 > (600-5):
                valid_detection = False
            
            if valid_detection:
                people_count += 1
                
                # Calculate center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Color based on confidence (green = high, yellow = medium, red = low)
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw rectangle for all people in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(frame, f'Person {confidence:.2f}', (x1, y1-10), 1, 1)
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Debug info (optional)
                if frame_count % 30 == 0:  # Print debug info every 30 frames
                    print(f"Person: conf={confidence:.2f}, area={bbox_area:.0f}, ratio={aspect_ratio:.2f}")
    
    # Display counting information on the frame
    cvzone.putTextRect(frame, f'Total People in Frame: {people_count}', (20, 50), 2, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'Frame: {frame_count}', (20, 100), 1, 1, colorR=(255, 0, 0))
    #cvzone.putTextRect(frame, f'Detection Settings', (20, 140), 1, 1, colorR=(255, 255, 255))
    #cvzone.putTextRect(frame, f'Conf: {YOLO_CONF} | Area: {MIN_AREA}-{MAX_AREA}', (20, 160), 1, 1, colorR=(255, 255, 255))
    
    # Optional: Print to console every 30 frames
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: {people_count} people detected in entire frame")
    
    cv2.imshow("RGB", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
