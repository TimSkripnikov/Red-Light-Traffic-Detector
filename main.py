from detector.detect_cars import CarDetector
import cv2

detector = CarDetector()

cap = cv2.VideoCapture("/home/artem/Documents/2_course/Python/project1/Red-Light-Traffic-Detector/data/videos/video_1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detector.detect(frame)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
