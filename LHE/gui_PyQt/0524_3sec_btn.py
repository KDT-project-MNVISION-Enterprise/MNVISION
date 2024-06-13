import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime
import time
import sys
import threading

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/Video.ui")[0] # Qt designer에서 작성한 ui 파일 불러오기
ort_session = YOLO('MNVISION/LHE/gui_PyQt/best.onnx') # 모델 파일 불러오기

class Video(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        self.cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.update_frame)  # 타이머가 만료될 때마다 update_frame 함수 호출
        self.timer.start(30)  # 30ms 간격으로 타이머 설정

        # 프레임을 저장할 리스트 생성
        self.fps_list = []

        # 새 창 띄우기 버튼
        # QDialog 설정 -> 새 창 띄울 때 사용.
        self.btn_play_3sec.clicked.connect(self.dialog_open)

    def update_frame(self):
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap.release()
            return
        self.show_img(self.frame)
        self.save_frames(self.fps_list)

    def save_frames(self, fps_list) :
        self.fps_list.append(self.frame)
        if len(fps_list) == 90 :
            del self.fps_list[0]

    def play_saved_frames(self):
        for fm in self.fps_list:
            print(fm)
            self.show_frame(fm)
            QCoreApplication.processEvents()  # UI 갱신을 위해 이벤트 처리

    def dialog_open(self) :
        self.dialog = QDialog()
        self.dialog.setWindowTitle('3초 전 영상 재생')
        self.play_frame_view = QGraphicsView(self.dialog)
        self.scene3 = QGraphicsScene()
        self.play_frame_view.setScene(self.scene3)
        
        # 다이얼로그 레이아웃 설정
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(self.play_frame_view)
        self.dialog.setLayout(dialog_layout)
        
        self.dialog.show()
        self.play_saved_frames()

    def show_img(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.on_air_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene2.clear()
        self.scene2.addPixmap(scaled_pixmap)
        self.on_air_camera.fitInView(self.scene2.itemsBoundingRect(), Qt.KeepAspectRatio)

    def show_frame(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.on_air_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene3.clear()
        self.scene3.addPixmap(scaled_pixmap)
        self.play_frame_view.fitInView(self.scene3.itemsBoundingRect(), Qt.KeepAspectRatio)

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
    myWindow = Video() 
    myWindow.show()
    sys.exit(app.exec_())