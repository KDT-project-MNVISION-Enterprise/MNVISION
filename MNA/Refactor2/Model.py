import cv2
import numpy as np
from ultralytics import YOLO

class VideoModel:
    def __init__(self, video_path, yolo_model_path):
        self.cap = cv2.VideoCapture(video_path)
        self.yolo_model = YOLO(yolo_model_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def detect_objects(self, frame):
        results = self.yolo_model(frame)
        detected_frame = results[0].plot()
        return detected_frame

    def transform_time(self, time_input):
        if isinstance(time_input, float):
            return f"{int(time_input // 3600):02}:{int((time_input % 3600) // 60):02}:{int(time_input % 60):02}"
        elif isinstance(time_input, str) and len(time_input.split(':')) == 3:
            hours, minutes, seconds = map(int, time_input.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError("Invalid input format. Provide an integer or a 'HH:MM:SS' string.")

    def set_frame_position(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def release(self):
        self.cap.release()