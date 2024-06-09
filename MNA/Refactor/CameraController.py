import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO

class CameraController(QObject):
    def __init__(self, main_window):
        super().__init__()  # QObject 초기화
        self.main_window = main_window
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit()
        self.ort_session = YOLO('Composite/best.onnx')
        self.model_detection = False
        self.setup_signals()
        self.timer = QTimer(self)  # 부모 객체로 self 전달
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def setup_signals(self):
        self.main_window.on_air_camera.setScene(QGraphicsScene())

    # 다른 메서드들...

    def toggle_model_detection(self):
        self.model_detection = not self.model_detection
        self.main_window.btn_start_detection.setText("모델 적용 중" if self.model_detection else "탐지 시작")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.model_detection:
                results = self.ort_session(frame)
                frame = results[0].plot()
            self.show_frame(frame)

    def show_frame(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.main_window.on_air_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.main_window.on_air_camera.scene().clear()
        self.main_window.on_air_camera.scene().addPixmap(scaled_pixmap)
        self.main_window.on_air_camera.fitInView(self.main_window.on_air_camera.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def release_resources(self):
        self.cap.release()