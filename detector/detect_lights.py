import cv2
import numpy as np

def detect_traffic_light_color(frame, mask_path):
    """
    Determines the current traffic signal on the frame using a mask.
    
    :param frame: Frame from video stream (BGR)
    :param mask_path: Path to PNG mask for traffic light area (white rectangle).
    :return: 'red', 'green' or 'none'
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("The mask does not contain contouring.")
        return "none"

    x, y, w, h = cv2.boundingRect(contours[0])
    roi_cropped = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, (40, 50, 50), (85, 255, 255))

    red_score = cv2.countNonZero(red_mask)
    green_score = cv2.countNonZero(green_mask)

    threshold = 30

    if red_score > green_score and red_score > threshold:
        return "red"
    elif green_score > threshold:
        return "green"
    else:
        return "none"
