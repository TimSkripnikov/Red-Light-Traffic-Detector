import cv2
import numpy as np
from detector.detect_cars import CarDetector
from detector.detect_lights import detect_traffic_light_color

class ViolationChecker:
    def __init__(self, traffic_light_mask, stop_line_mask, car_model_path="model/yolov8n_car.pt", conf_threshold=0.3):

        self.car_detector = CarDetector(model_path=car_model_path, conf_threshold=conf_threshold)
        self.traffic_light_mask = traffic_light_mask
        self.stop_line_mask = stop_line_mask

    def check_violation(self, frame):
        """
        Checks if the car violated the rules by running a red traffic light.

        :param frame: Frame (BGR, np.ndarray)
        :return: True if violation, False if not
        """

        light_color = detect_traffic_light_color(frame, self.traffic_light_mask)
        #print(f"Color of traffic lights: {light_color}")  

        if light_color == "green":
            return False
        
        car_boxes = self.car_detector.detect(frame)
       # print(f"Count of cars: {len(car_boxes)}")  

        if not car_boxes:
            return False

        stop_line = cv2.imread(self.stop_line_mask, cv2.IMREAD_GRAYSCALE)
        _, stop_line_bin = cv2.threshold(stop_line, 1, 255, cv2.THRESH_BINARY)

        for (x1, y1, x2, y2) in car_boxes:
            
            car_roi = frame[y1:y2, x1:x2]
            car_mask = cv2.inRange(car_roi, (0, 0, 0), (255, 255, 255)) 

            
            intersection = cv2.bitwise_and(car_mask, stop_line_bin[y1:y2, x1:x2])  
            if np.sum(intersection) > 0: 
                return True
        
        return False
