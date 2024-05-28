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

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/Video3.ui")[0] # Load the UI file
ort_session = YOLO('MNVISION/LHE/gui_PyQt/best.onnx') # Load the model

class WindowClass(QMainWindow, form_class): 
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Register scene for graphicsView
        self.scene = QGraphicsScene() 
        self.graphicsView.setScene(self.scene) 

        # Connect buttons to functions
        self.btn_pause.clicked.connect(self.start_video) 
        self.Video_upload.clicked.connect(self.btn_fun_FileLoad)
        self.btn_stop_start.clicked.connect(self.btn_change_pause)
        
        # Initialize variables
        self.filepath=""
        self.cap=None
        self.sleep_ms=0
        self.is_playing=True
    
    # Load video file
    def btn_fun_FileLoad(self):        
        self.filepath, _ = QFileDialog.getOpenFileName(self, 'Open file', './', "Video files (*.mp4 *.avi)")        
        if self.filepath:
            print(self.filepath)
            self.cap = cv2.VideoCapture(self.filepath) 
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) 
            self.sleep_ms = int(np.round((1 / self.fps) * 1000)) 

            # Display video information when file is loaded
            self.display_video_info()
    
    # Start video playback
    def start_video(self):
        while True:
            if self.is_playing:
                ret, frame = self.cap.read() 
                results = ort_session(frame) 
                frame = results[0].plot()

                qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888) 
                pixmap = QPixmap(qt_image) 
                scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.scene.clear() 
                self.scene.addPixmap(scaled_pixmap) 
                self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio) 
            key = cv2.waitKey(30)

        self.cap.release() 

    # Pause/play video
    def btn_change_pause(self):
        self.is_playing = not self.is_playing

    # Display video information
    def display_video_info(self):
        if self.cap is not None:
            self.filename = os.path.basename(self.filepath)
            self.length=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps=int(self.cap.get(cv2.CAP_PROP_FPS))
            self.video_length = datetime.timedelta(seconds=int(self.length/self.fps))
            info_text = f"파일명: {self.filename}\n프레임: {self.length}\n크기: {self.width} x {self.height}\n영상 길이: {self.video_length}\nfps: {self.fps}"
            self.Video_info_text.setText(info_text)

    # Close confirmation dialog
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