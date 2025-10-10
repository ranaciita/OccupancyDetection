import datetime
from datetime import datetime
import time
import file_save_handler
import numpy as np
import cv2

# test class tests saving txt and image logs in /results using file_save_handler

frame_id = 1
frame_count = 0

# saving files using file_save_handler
logClusterID = datetime.now().strftime("ID_%Y%m%d_%H%M%S_%f")
fsh = file_save_handler.file_save_handler()
fsh.create_new_folder(logClusterID)

fsh.add_log_to_txt(f"Log Cluster: {logClusterID}")
fsh.add_log_to_txt("------------------------------------------------")
fsh.add_log_to_txt("SETTINGS:")
fsh.add_log_to_txt("Minimum Confidence: 0.15")
fsh.add_log_to_txt("Minimum Area: 200")
fsh.add_log_to_txt("Aspect Ratio Range: 0.15 - 10.0")
fsh.add_log_to_txt("Frame Resolution: 1020 x 600")
fsh.add_log_to_txt("------------------------------------------------")

for x in range(15):
    
    # text log example
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    process_time = 0.123
    people_count = 5
    
    fsh.add_log_to_txt(f"ID: {frame_id} | frame: {frame_count} | start: {timestamp_start} | end: {timestamp_end} | process: {process_time:.3f}s | people detected: {people_count}")

    # image log example
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    color = (0, 255 // (x + 1), 255) 
    cv2.rectangle(frame, (50, 50), (250, 150), color, -1)
    image_name = f"frame_{x+1}.jpg"

    fsh.add_image_log(frame, image_name)

    # next frame
    frame_id += 1
    frame_count += 150
    time.sleep(1)
