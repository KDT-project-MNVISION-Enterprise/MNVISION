import datetime
import time
import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import QDesktopWidget
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


class LogController:
    def __init__(self, main_window):
        self.main_window = main_window
        self.fps_list = []
        self.dialog = None
        self.setup_log_text()

    def setup_log_text(self):
        for i in range(1, 2):
            self.main_window.Log_text_2.addItem(f"위험상황 포착")
        for i in range(1, 10):
            self.main_window.Log_text.addItem(f"00:0{i}:00")
        self.main_window.Log_text.itemClicked.connect(self.seek_video)

    def show_dialog(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle('3초 전 영상 재생')
        play_frame_view = QGraphicsView(self.dialog)
        scene = QGraphicsScene()
        play_frame_view.setScene(scene)

        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(play_frame_view)
        self.dialog.setLayout(dialog_layout)

        message_label = QLabel("동영상 저장 중", self.dialog)
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet("QLabel {font-size: 24px; font-weight: bold; }")
        dialog_layout.addWidget(message_label)

        screen = QDesktopWidget().screenGeometry()
        width = screen.width() // 2
        height = screen.height() // 2
        self.dialog.resize(width, height)

        self.dialog.show()
        self.save_and_play_frames(play_frame_view, scene, message_label)

    def save_and_play_frames(self, play_frame_view, scene, message_label):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        output_path = f'output_video_{timestamp}.mp4'
        frame_delay = 1 / 30
        if len(self.fps_list) == 0:
            print("Error: No frames to save.")
            return
        frame_height, frame_width, _ = self.fps_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
        start_time = time.time()
        for fm in self.fps_list:
            out.write(fm)
            self.show_frame(play_frame_view, scene, fm)
            QCoreApplication.processEvents()
            time.sleep(frame_delay)
            if time.time() - start_time >= 6:
                break
        out.release()
        message_label.setText("동영상 저장 완료")

    def show_frame(self, element, scene, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(element.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scene.clear()
        scene.addPixmap(scaled_pixmap)
        element.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def seek_video(self, item):
        transform_time = self.main_window.video_controller.transform_time(item.text())
        self.main_window.video_controller.cap.set(cv2.CAP_PROP_POS_FRAMES, transform_time * self.main_window.video_controller.fps)
        self.main_window.Video_bar.setValue(int(transform_time * self.main_window.video_controller.fps / self.main_window.video_controller.frame_count * 100))