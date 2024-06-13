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

# Tab1과 Tab3 클래스가 정의된 파일 가져오기
from tab1 import Tab1
from tab3 import Tab3

ort_session = YOLO('MNVISION/LHE/gui_PyQt/Composite/best.onnx')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("MNVISION/LHE/gui_PyQt/Composite/Video4.ui", self)

        # 'on_air_detection' QTabWidget 객체 찾기
        self.on_air_detection = self.findChild(QTabWidget, 'on_air_detection')

        # Tab3 인스턴스 생성
        self.tab3_instance = Tab3()


def gui():
    # QMainWindow 인스턴스 생성
    main_window = MainWindow()
    main_window.setWindowTitle("Main Window")

    # 생성한 QMainWindow 반환
    return main_window

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = gui()
    myWindow.show()
    sys.exit(app.exec_())