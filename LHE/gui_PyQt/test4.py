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
import threading

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/Video.ui")[0] # Qt designer에서 작성한 ui 파일 불러오기
ort_session = YOLO('MNVISION/LHE/gui_PyQt/best.pt') # 모델 파일 불러오기

class WindowClass2(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        self.video_thread = VideoThread()
        self.video_thread.frame_update.connect(self.update_frame)
        self.video_thread.start()

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        results = ort_session(frame)
        frame = results[0].plot()
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888) 
        pixmap = QPixmap(qt_image) 
        scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene2.clear() 
        self.scene2.addPixmap(scaled_pixmap) 
        self.graphicsView.fitInView(self.scene2.itemsBoundingRect(), Qt.KeepAspectRatio) 

    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)
        if re == QMessageBox.Yes:
            QCloseEvent.accept()
            sys.exit()
        else:
            QCloseEvent.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass2() 
    myWindow.show()
    sys.exit(app.exec_())