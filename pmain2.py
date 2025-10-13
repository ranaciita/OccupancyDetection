# imports
from ultralytics import YOLO 
import cv2 # used for video/image processing and showing frames
import cvzone # for showing information on the frames
import pandas as pd # for easier handling the data results from YOLO
import file_save_handler
import time
from datetime import datetime

# ------------------------------------------------------------
# ---------------------- 1. Setting up -----------------------
# ------------------------------------------------------------

# we are using YOLO version 11m
model = YOLO("yolo11m.pt")

# the camera is hosted in this URL (IP might change each time)
# "root" = username
# "YoloTracking" = password
# /axis-media/media.amp = axis fixed path
IP = "169.254.41.208"
CAM_URL = f"rtsp://root:YoloTracking@{IP}/axis-media/media.amp"

# load the COCO classes + save in an array
# (the file coco.txt contains different classes that YOLO can detect)
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# specify the detection settings
MIN_CONFIDENCE = 0.15
MIN_AREA = 200
MAX_AREA = float('inf')
ASPECT_RATIO_MIN = 0.15
ASPECT_RATIO_MAX = 10.0

# frame resolution
FRAME_WIDTH = 1020
FRAME_HEIGHT = 600

# var used to count frames and ID
frame_count = 0
frame_id = 0

# saving files using file_save_handler
logClusterID = datetime.now().strftime("ID_%Y%m%d_%H%M%S_%f")
fsh = file_save_handler.file_save_handler()
fsh.create_new_folder(logClusterID)

fsh.add_log_to_txt(f"Log Cluster: {logClusterID}")
fsh.add_log_to_txt("------------------------------------------------")
fsh.add_log_to_txt("SETTINGS:")
fsh.add_log_to_txt("Minimum Confidence: " + str(MIN_CONFIDENCE))
fsh.add_log_to_txt("Minimum Area: " + str(MIN_AREA))
fsh.add_log_to_txt("Aspect Ratio Range: " + str(ASPECT_RATIO_MIN) + " - " + str(ASPECT_RATIO_MAX))
fsh.add_log_to_txt("Frame Resolution: " + str(FRAME_HEIGHT) + " x " + str(FRAME_WIDTH))
fsh.add_log_to_txt("------------------------------------------------")

# ------------------------------------------------------------
# ---------------- 2. Processing the stream ------------------
# ------------------------------------------------------------

# continuous loop - processes each frame until stopped.
# ...for each result when running YOLO on the RTSP stream:
for results in model.predict(CAM_URL, stream=True, conf=0.15, iou=0.45, verbose=False):
    
    # increment frame count
    frame_count += 1

    # only process every 150th frame, otherwise continue to next iteration
    if frame_count % 3 != 0:
        continue
    
    # increment the id
    frame_id += 1

    # for the log:
    # record start time of processing this frame
    frame_start_time = time.time()
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # get a copy of the actual/original frame image from the stream:
    frame = results.orig_img.copy()

    # get the detected objects and their bounding boxes
    # converts the results to a pandas dataframe -> easier to handle 
    a = results.boxes.data
    px = pd.DataFrame(a).astype("float")

    # var used to count people (in this specific frame)
    people_count = 0

    # loop through the detected objects (stored in px pandas dataframe)
    for index, row in px.iterrows():

        # extract the data from each index
        x1, y1, x2, y2 = map(int, row[:4]) # the bounding box coordinates
        confidence = float(row[4]) # confidence of this detection
        cls_id = int(row[5]) # the class id (for example 0 for "person")
        c = class_list[cls_id] # convert cls_id to its corresponding string representation

        # only show objects of class "person" that meet minimum confidence requirement
        if c == "person" and confidence > MIN_CONFIDENCE:
            
            # calculate aspect ratio + area of the bounding box
            width = x2 - x1
            height = y2 - y1
            bbox_area = width * height
            aspect_ratio = height / width if width > 0 else 0

            # bool used to determine if the detection is valid
            valid_detection = True

            # if the detection doesn't meet the defined critera, mark invalid
            if bbox_area < MIN_AREA:
                valid_detection = False
            if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                valid_detection = False
            if width < 8 or height < 16:  # for small/partial detections
                valid_detection = False

            # if the detection is valid, draw it on the frame
            if valid_detection:
                people_count += 1

                # choose the frame color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0) #green
                elif confidence > 0.5:
                    color = (0, 255, 255) #yellow
                else:
                    color = (0, 0, 255) #red

                # draw the bounding box + info (using cv2/cvzone) on the image
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(frame, f'Person {confidence:.2f}', (x1, y1-10), scale=2, thickness=2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

# ------------------------------------------------------------
# --------------- 3. Overlay Info Preferences ----------------
# ------------------------------------------------------------

    # display total people count + frame number on the top-left corner
    cvzone.putTextRect(frame, f'Total People: {people_count}', (20, 50), scale=3, thickness=3, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'Frame: {frame_count}, ID: {frame_id}', (20, 120), scale=2.5, thickness=2, colorR=(255, 0, 0))
    # display frame ID (corresponding to the log ID) on the image
    # cvzone.putTextRect(frame, f'ID: {frame_id}', (20, 120), scale=2.5, thickness=2, colorR=(255, 0, 0))


    # calculate total frame processing time
    frame_end_time = time.time()
    timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    process_time = round(frame_end_time - frame_start_time, 3)

    # resize the frame + open a window to show the stream 
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.imshow("RTSP Camera", frame)

    # log info to text file
    log_line = (
        f"ID: {frame_id} | frame: {frame_count} | start: {timestamp_start} | "
        f"end: {timestamp_end} | process: {process_time:.3f}s | people detected: {people_count}"
    )
    fsh.add_log_to_txt(log_line)

    # save the frame image
    image_name = f"{frame_id}.jpg"
    fsh.add_image_log(frame, image_name)

    # break loop if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# clean terminate application after breaking out the loop
cv2.destroyAllWindows()
