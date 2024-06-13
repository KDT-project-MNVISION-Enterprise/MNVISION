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


class WindowClass2(QMainWindow, form_class): # c
    def __init__(self):
        """
        클래스 초기화 함수.

        QMainWindow와 form_class를 상속받아 윈도우 클래스를 초기화한다.

        :return: None
        """
        super().__init__()
        self.setupUi(self)

        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)
        self.start_detection()
        self.btn_start_detection.clicked.connect(self.start_detection)

    def start_detection(self):
        """
        카메라 실시간 감지 함수.

        :return: None
        """
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened() :
            ret, self.frame = self.cap.read()
            results = ort_session(frame)
            frame = results[0].plot()
            qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888) 
            pixmap = QPixmap(qt_image) 
            scaled_pixmap = pixmap.scaled(self.on_air_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.scene2.clear() 
            self.scene2.addPixmap(scaled_pixmap) 
            self.on_air_camera.fitInView(self.scene2.itemsBoundingRect(), Qt.KeepAspectRatio) 
            key = cv2.waitKey(30)
        self.cap.release()


    def closeEvent(self, QCloseEvent):
        """
        창을 닫을 때의 이벤트 핸들러 함수.

        사용자에게 종료 여부를 확인하고, Yes 버튼을 클릭하면 종료하고, No 버튼을 클릭하면 종료하지 않는다.

        :param QCloseEvent: 닫기 이벤트 객체
        :return: None
        """
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)  # 종료 여부 확인 다이얼로그 표시
        if re == QMessageBox.Yes:  # Yes 버튼을 클릭한 경우
            QCloseEvent.accept()  # 종료
            sys.exit()
        else:  # No 버튼을 클릭한 경우
            QCloseEvent.ignore()  # 종료하지 않음.

    def start_detection(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass2() 
    myWindow.show()
    sys.exit(app.exec_())

