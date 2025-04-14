from detector.violation_checker import ViolationChecker
import cv2

traffic_light_mask = "data/masks/mask_lights_1.png"
stop_line_mask = "data/masks/mask_stop_line_1.1.png"

violation_checker = ViolationChecker(traffic_light_mask, stop_line_mask)

cap = cv2.VideoCapture("data/videos/video_1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    violation = violation_checker.check_violation(frame)
    
    if violation:
        cv2.putText(frame, "Violation: Ran a red light!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
