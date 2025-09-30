import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
#from test1 import process_frame
import os
from tracker import*
from datetime import datetime

model = YOLO("yolov10s.pt")  

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#cap = cv2.VideoCapture('tr1.mp4')
cap = cv2.VideoCapture('rtsp://root:YoloTracking@169.254.138.29/axis-media/media.amp')

# Check if video file exists
if not cap.isOpened():
    print("Error: Could not open video file 'tr1.mp4'")
    print("Please download the video file from the link in vid.txt")
    exit()

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker=Tracker()
count = 0

# Make the detection line longer and more horizontal
# Old area: [(324, 313), (283, 374), (854, 392), (864, 322)]
# New longer line across most of the frame width
#area = [(100, 350), (50, 400), (950, 400), (950, 350)]

# Detection area: a single wide horizontal line (thin rectangle) covering the whole frame width
# Adjust y1 and y2 to set the vertical position and thickness of the line
line_y1 = 375  # vertical position (top of the line)
line_y2 = 385  # vertical position (bottom of the line, for thickness)
frame_width = 1020  # matches the resized frame width
area = [(0, line_y1), (0, line_y2), (frame_width, line_y2), (frame_width, line_y1)]
#area = [(400, 400), (400, 500), (600, 500), (600, 400)]

# Counting variables
people_entered = 0  # People going down (crossing line from top to bottom)
people_left = 0     # People going up (crossing line from bottom to top)
previous_positions = {}  # Store previous y-positions of tracked objects
crossed_objects = set()  # Objects that have already been counted

# Create directory for today's date
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
list1=[]
while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    #processed_frame, detected_label = process_frame(frame)
    
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        # Only add bounding boxes for 'person' class
        if c == 'person':
            list.append([x1, y1, x2, y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
    
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        
        # Check if person is crossing the detection area
        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        
        # Track direction and count entries/exits
        if id in previous_positions:
            prev_cy = previous_positions[id]
            current_cy = cy
            
            # Check if crossing the line (center of detection area is around y=375)
            line_y = 375  # Approximate center of our detection area
            
            if result >= 0:  # Person is in the detection area
                # Check if person crossed from top to bottom (entered)
                if prev_cy < line_y and current_cy >= line_y and id not in crossed_objects:
                    people_entered += 1
                    crossed_objects.add(id)
                    print(f"Person {id} ENTERED. Total entered: {people_entered}")
                
                # Check if person crossed from bottom to top (left)
                elif prev_cy > line_y and current_cy <= line_y and id not in crossed_objects:
                    people_left += 1
                    crossed_objects.add(id)
                    print(f"Person {id} LEFT. Total left: {people_left}")
        
        # Update previous position
        previous_positions[id] = cy
        
        if result>=0:
           # Only track people (not cars for people counting)
           if 'person' in c:  # Changed from 'car' to 'person' for people counting
                cvzone.putTextRect(frame, f'Person {id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
           else:     
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                
    

           

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    
    # Display counting information on the frame
    cvzone.putTextRect(frame, f'People Entered: {people_entered}', (20, 50), 2, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'People Left: {people_left}', (20, 100), 2, 2, colorR=(0, 0, 255))
    cvzone.putTextRect(frame, f'Net Count: {people_entered - people_left}', (20, 150), 2, 2, colorR=(255, 0, 0))
    
    # Draw direction arrows to show which way is "entered" vs "left"
    cv2.arrowedLine(frame, (400, 320), (400, 280), (0, 0, 255), 3, tipLength=0.1)  # Up arrow (LEFT)
    cv2.arrowedLine(frame, (450, 280), (450, 320), (0, 255, 0), 3, tipLength=0.1)  # Down arrow (ENTERED)
    cvzone.putTextRect(frame, 'LEFT', (370, 260), 1, 1, colorR=(0, 0, 255))
    cvzone.putTextRect(frame, 'ENTERED', (420, 260), 1, 1, colorR=(0, 255, 0))
    
    cv2.imshow("RGB", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()