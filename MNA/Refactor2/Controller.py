from Model import VideoModel
from View import VideoView
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import time
import os
import threading

class VideoController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.view = VideoView()
        self.model = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.setup_signals()

    def setup_signals(self):
        self.view.load_video.connect(self.load_video)
        self.view.start_video.connect(self.start_video)
        self.view.toggle_model_detection.connect(self.toggle_model_detection)
        self.view.toggle_video_playback.connect(self.toggle_video_playback)
        self.view.forward_video.connect(self.forward_video)
        self.view.backward_video.connect(self.backward_video)
        self.view.seek_video.connect(self.seek_video)

    def load_video(self, file_path):
        self.model = VideoModel(file_path, 'Composite/best.onnx')
        self.view.Total_length.setText(self.model.transform_time(self.model.duration))
        self.view.show_video_controls()

    def start_video(self):
        self.timer.start(30)
        self.is_playing = True

    def toggle_model_detection(self):
        self.view.btn_start_detection.setText("모델 적용 중" if self.model_detection else "탐지 시작")
        self.model_detection = not self.model_detection

    def update_frame(self):
        if self.is_playing:
            ret, frame = self.model.get_frame()
            if ret:
                if self.model_detection:
                    frame = self.model.detect_objects(frame)
                self.view.update_frame(frame)
            else:
                self.timer.stop()
                self.is_playing = False


    
    def toggle_video_play(self) :
        if(self.is_playing):
            self.is_playing = not self.is_playing
            if self.is_playing:
                self.timer.start(30)
                icon = QIcon('Video/icon/play.png')
            else:
                self.timer.stop()
                icon = QIcon('Video/icon/stop-button.png')
                self.view.btn_stop_start.setIcon(icon)
    
    def forward_video(self):
        current_frame = self.model.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = current_frame + int(100/6 * self.model.fps)
        if new_frame >= self.model.frame_count:
            new_frame = self.model.frame_count - 1
        self.model.set_frame_position(new_frame)
        self.view.Video_bar.setValue(int(new_frame / self.model.frame_count * 100))

    def backward_video(self):
        current_frame = self.model.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = current_frame - int(50/6 * self.model.fps)
        if new_frame < 0:
            new_frame = 0
        self.model.set_frame_position(new_frame)
        self.view.Video_bar.setValue(int(new_frame / self.model.frame_count * 100))

    def seek_video(self, position):
        new_frame = int(position * self.model.frame_count / 100)
        self.model.set_frame_position(new_frame)
        duration = new_frame / self.model.fps
        self.view.Current.setText(self.model.transform_time(duration))

    def run(self):
        self.view.show()
        sys.exit(self.app.exec_())
if __name__ == "__main__":
    controller = VideoController()
    controller.run()