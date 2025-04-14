import cv2
import os

video_path = '/home/artem/Documents/2_course/Python/project1/Red-Light-Traffic-Detector/data/videos/video_1.mp4'
number_of_video = 1
frame_number_to_save = 10


output_dir = '/home/artem/Documents/2_course/Python/project1/Red-Light-Traffic-Detector/data/frames'

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_to_save)
ret, frame = cap.read()

if ret:
    output_filename = os.path.join(output_dir, f'frame_{number_of_video}.jpg')
    cv2.imwrite(output_filename, frame)
else:
    print("Can't get frame.")

cap.release()
