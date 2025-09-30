import cv2
import numpy as np

def process_frame(frame):
    """
    Process frame to detect red color (traffic light detection)
    Returns the processed frame and detected label
    """
    # Convert frame to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Define range for red color in HSV,
#Red color has two ranges in HSV,
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

#Create masks for red color,
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

#Apply morphological operations to clean up the mask,
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPHOPEN, kernel)

#Find contours in the red mask,
    contours,  = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_label = "GREEN"  # Default to green

#Check if significant red areas are detected,
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold for red detection
            detected_label = "RED"
            # Draw contour on frame for visualization (optional)
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
            break

    return frame, detected_label