import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import numpy as np
from ultralytics import YOLO
import os

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/Video2.ui")[0]
ort_session = YOLO('MNVISION/LHE/gui_PyQt/best.onnx')

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scene = QGraphicsScene() 
        self.graphicsView.setScene(self.scene)
        
        self.btn_stop_start.clicked.connect(self.toggle_video)
        self.pushButton_7.clicked.connect(self.btn_fun_FileLoad)
        
        self.filepath=""
        self.is_playing = False

    def btn_fun_FileLoad(self):        
        fname=QFileDialog.getOpenFileName(self, 'Open file', './')        
        print(fname[0])
        print(fname[1])
        self.filepath=fname[0]

    def toggle_video(self):
        self.is_playing = not self.is_playing # 상태 토글
        if self.is_playing:
            self.start_video() # 동영상 재생 시작

    def start_video(self):
        print(self.filepath)
        cap = cv2.VideoCapture(rf"{self.filepath}") 
        fps = cap.get(cv2.CAP_PROP_FPS) 
        sleep_ms = int(np.round((1 / fps) * 1000)) 
    
        while True:
            if self.is_playing:
                ret, frame = cap.read() 
                if not ret: 
                    break

                results = ort_session(frame) 
                frame = results[0].plot()

                qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888) 
                pixmap = QPixmap(qt_image) 

                scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.scene.clear() 
                self.scene.addPixmap(scaled_pixmap) 
                self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio) 

                key = cv2.waitKey(sleep_ms)

                if key == ord('q'):
                    break

        cap.release() 

    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                    QMessageBox.Yes|QMessageBox.No) 

        if re == QMessageBox.Yes: 
            QCloseEvent.accept() 

        else: 
            QCloseEvent.ignore() 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass() 
    myWindow.show()
    sys.exit(app.exec_())