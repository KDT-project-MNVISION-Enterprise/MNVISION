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

form_class = uic.loadUiType("Video/Video.ui")[0] # Qt designer에서 작성한 ui 파일 불러오기
ort_session = YOLO('Streaming/best.onnx') # 모델 파일 불러오기

class WindowClass(QMainWindow, form_class): # 
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scene = QGraphicsScene() 
        self.graphicsView.setScene(self.scene)
        self.btn_pause.clicked.connect(self.start_video)
        self.Video_upload.clicked.connect(self.btn_fun_FileLoad)
        self.btn_stop_start.clicked.connect(self.btn_change_pause)
        self.btn_forward.clicked.connect(self.forward)
        self.btn_prev.clicked.connect(self.backward)
        self.Video_bar.valueChanged.connect(self.slider_moved)
        for i in range(1,10):
            self.Log_text.addItem(f"00:0{i}:00")
        self.Log_text.itemClicked.connect(self.item_clicked)
        self.filepath=""
        self.cap=None
        self.sleep_ms=0
        self.is_playing=False
        self.frame_count =None
        self.duration = None
        self.current_frame=0
        self.fps=None

    def btn_fun_FileLoad(self):        
        fname=QFileDialog.getOpenFileName(self, 'Open file', './')        
        self.filepath=fname[0]
        self.cap = cv2.VideoCapture(rf"{self.filepath}") 
        if not self. cap.isOpened():
            print("Error: Could not open video.")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) 
        self.sleep_ms = int(np.round((1 / fps) * 1000)) 
        ret, frame = self.cap.read()
        self.current_frame += 1
        self.show_img(frame)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.Total_length.setText(self.transform_time(self.duration))
        self.display_video_info()

    def transform_time(self, time_input):
        if isinstance(time_input, float):
            # 입력값이 정수인 경우 시:분:초 형식으로 변환
            seconds = time_input
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        elif isinstance(time_input, str) and len(time_input.split(':')) == 3:
            # 입력값이 "시:분:초" 형식의 문자열인 경우 초 단위로 변환
            time_parts = time_input.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            return total_seconds
        else:
            raise ValueError("Invalid input format. Provide an integer or a 'HH:MM:SS' string.")
        

    def start_video(self):
        while True:
            if self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    self.is_playing = False
                    break
                self.current_frame += 1
                self.duration = self.current_frame / self.fps
                self.Current.setText(self.transform_time(self.duration))
                results = ort_session(frame)
                frame = results[0].plot()
                self.show_img(frame)
            key = cv2.waitKey(30)
        self.cap.release() # 코드가 끝나면 메모리 해제
    
    def show_img(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888) 
        pixmap = QPixmap(qt_image) 
        scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def item_clicked(self, item):
        transform_time = self.transform_time(item.text())
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, transform_time*self.fps)

    def slider_moved(self, position):
        self.current_frame = int(position*self.frame_count/100)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.Current.setText(self.transform_time(self.duration))

    def btn_change_pause(self):
        
        icon = QIcon('Video/icon/play.png')  # 이미지 파일 경로를 설정
        self.btn_stop_start.setIcon(icon)
        self.is_playing = not self.is_playing

    
    def forward(self):
        self.current_frame += int(10 * self.fps)
        if self.current_frame >= self.frame_count:
            self.current_frame = self.frame_count - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame)
        
    def backward(self):
        self.current_frame -= int(10 * self.fps)
        if self.current_frame < 0:
            self.current_frame = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
    
    
    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                    QMessageBox.Yes|QMessageBox.No) # 창 이름 : 종료 확인 / 메시지 : 종료하시겠습니까? / 버튼 : yes 또는 no

        if re == QMessageBox.Yes: # yes 선택 시
            QCloseEvent.accept() # 종료
        else: # no 선택 시
            QCloseEvent.ignore() # 종료 하지 않음.
            
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

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass() 
    myWindow.show()
    sys.exit(app.exec_())
