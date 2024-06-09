import cv2
import numpy as np
import datetime
import os
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
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
    def __init__(self, main_window):
        self.main_window = main_window
        self.cap = None
        self.timer_video = None
        self.is_playing = False
        self.setup_signals()

    def setup_signals(self):
        self.main_window.graphicsView.setScene(QGraphicsScene())

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, 'Open file', './', 'Video files (*.mp4 *.avi)')
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.main_window.Total_length.setText(self.transform_time(self.duration))
        self.main_window.show_video_controls()

    def transform_time(self, time_input):
        if isinstance(time_input, float):
            return f"{int(time_input // 3600):02}:{int((time_input % 3600) // 60):02}:{int(time_input % 60):02}"
        elif isinstance(time_input, str) and len(time_input.split(':')) == 3:
            hours, minutes, seconds = map(int, time_input.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError("Invalid input format. Provide an integer or a 'HH:MM:SS' string.")

    def start_video(self):
        self.timer_video = QTimer(self.main_window)
        self.timer_video.timeout.connect(self.process_video)
        self.timer_video.start(30)
        self.is_playing = True

    def process_video(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                self.timer_video.stop()
                self.cap.release()
                return
            self.update_video_info(frame)
            self.show_frame(frame)

    def update_video_info(self, frame):
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.duration = current_frame / self.fps
        self.main_window.Current.setText(self.transform_time(self.duration))

    def show_frame(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.main_window.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.main_window.graphicsView.scene().clear()
        self.main_window.graphicsView.scene().addPixmap(scaled_pixmap)
        self.main_window.graphicsView.fitInView(self.main_window.graphicsView.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def toggle_video_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.start_video()
            icon = QIcon('Video/icon/play.png')
        else:
            self.timer_video.stop()
            icon = QIcon('Video/icon/stop-button.png')
        self.main_window.btn_stop_start.setIcon(icon)

    def forward_video(self):
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = current_frame + int(100/6 * self.fps)
        if new_frame >= self.frame_count:
            new_frame = self.frame_count - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.main_window.Video_bar.setValue(int(new_frame / self.frame_count * 100))

    def backward_video(self):
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = current_frame - int(50/6 * self.fps)
        if new_frame < 0:
            new_frame = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.main_window.Video_bar.setValue(int(new_frame / self.frame_count * 100))

    def seek_video(self, position):
        new_frame = int(position * self.frame_count / 100)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.update_video_info(None)

    def release_resources(self):
        if self.cap is not None:
            self.cap.release()