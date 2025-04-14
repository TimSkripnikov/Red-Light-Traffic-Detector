from ultralytics import YOLO
import numpy as np

class CarDetector:
    def __init__(self, model_path="/home/artem/Documents/2_course/Python/project1/Red-Light-Traffic-Detector/model/yolov8n_car.pt", conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Детектирует автомобили на изображении.

        :param frame: Кадр (BGR, np.ndarray)
        :return: Список bounding boxes: [(x1, y1, x2, y2), ...]
        """
        results = self.model(frame)[0]
        boxes = []

        for box in results.boxes:
            conf = box.conf.item()
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))

        return boxes
