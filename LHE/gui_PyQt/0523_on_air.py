import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO
import numpy as np
import os
import datetime
import time
import threading

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/practice/Video.ui")[0]
ort_session = YOLO('MNVISION/LHE/gui_PyQt/practice/best.onnx')


class WindowClass2(QMainWindow, form_class): # c
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)
        self.btn_start_detection.clicked.connect(self.start_detection_from_button)
        # setText
        self.cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.update_frame)  # 타이머가 만료될 때마다 update_frame 함수 호출
        self.timer.start(30)  # 30ms 간격으로 타이머 설정

        self.detection_flag = False  # 탐지 플래그 초기화

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap.release()
            return

        if self.detection_flag:  # 탐지 플래그가 True이면 YOLO 모델 실행
            self.start_detection(frame)
        else:
            self.show_img(frame)

    def start_detection_from_button(self):
        self.detection_flag = not self.detection_flag  # 탐지 플래그를 True로 설정
        if self.detection_flag:
            self.btn_start_detection.setText("탐지 종료")  # 버튼 텍스트 변경
        else:
            self.btn_start_detection.setText("탐지 시작")

    def start_detection(self, frame):
        results = ort_session(frame)  # YOLO 객체 감지
        frame = results[0].plot()  # 결과 플롯
        self.show_img(frame)  # 이미지 표시

    def show_img(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.on_air_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene2.clear()
        self.scene2.addPixmap(scaled_pixmap)
        self.on_air_camera.fitInView(self.scene2.itemsBoundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)  # 종료 여부 확인 다이얼로그 표시
        if re == QMessageBox.Yes:  # Yes 버튼을 클릭한 경우
            QCloseEvent.accept()  # 종료
            sys.exit()
        else:  # No 버튼을 클릭한 경우
            QCloseEvent.ignore()  # 종료하지 않음.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass2() 
    myWindow.show()
    sys.exit(app.exec_())

