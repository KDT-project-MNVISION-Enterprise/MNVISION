# 탭 1에 들어가는 클래스 

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

# form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/Composite/Video4.ui")[0]
ort_session = YOLO('MNVISION/LHE/gui_PyQt/Composite/best.onnx')

class Tab1(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("MNVISION/LHE/gui_PyQt/Composite/Video4.ui", self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)


        self.btn_pause.clicked.connect(self.start_video)
        self.btn_pause.hide()
        self.Current.hide()
        self.Total_length.hide()
        self.Video_bar.hide()
        self.btn_forward.hide()
        self.btn_prev.hide()
        self.btn_stop_start.hide()
        self.Video_upload.clicked.connect(self.btn_fun_FileLoad)
        self.btn_stop_start.clicked.connect(self.btn_change_pause)
        self.btn_forward.clicked.connect(self.forward)
        self.btn_prev.clicked.connect(self.backward)
        self.Video_bar.valueChanged.connect(self.slider_moved)
        self.progressBar.setValue(0)      

        for i in range(1, 10):
            self.Log_text.addItem(f"00:0{i}:00")
        self.Log_text.itemClicked.connect(self.item_clicked)   


        self.timer = QTimer(self) 

        self.filepath = ""  
        self.cap = None  
        self.sleep_ms = 0  
        self.is_playing = True
        self.frame_count = None  
        self.duration = None  
        self.current_frame = 0  
        self.fps = None  
        self.progress_value = 0 
        self.flag=0 
        self.fps_list = []

    def start_video(self):
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(10)  
        self.btn_pause.hide()
        self.Current.show()
        self.Total_length.show()
        self.Video_bar.show()
        self.btn_forward.show()
        self.btn_prev.show()
        self.btn_stop_start.show()

        self.timer_video = QTimer(self)  # 비디오 처리용 타이머 생성
        self.timer_video.timeout.connect(self.process_video)  # 타이머가 만료될 때마다 process_video 메서드 호출
        self.timer_video.start(30)  # 30ms 간격으로 타이머 설정

    def process_video(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                self.timer_video.stop()  # 비디오 끝에 도달하면 타이머 중지
                self.cap.release()
                return
            self.current_frame += 1  # 현재 프레임 수 증가
            self.duration = self.current_frame / self.fps  # 재생 시간 업데이트
            self.Current.setText(self.transform_time(self.duration))  # 현재 재생 시간 업데이트
            results = ort_session(frame)  # YOLO 객체 감지
            frame = results[0].plot()  # 결과 플롯
            self.show_img(self.graphicsView, self.scene, frame)  # 이미지 표시

    def update_progress(self):
        self.progress_value += 1
        self.progressBar.setValue(self.progress_value)
        if self.progress_value >= 100:
            self.timer.stop()


    def btn_fun_FileLoad(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './','Video files (*.mp4 *.avi)')
        if fname[0]:
            self.filepath = fname[0]  
            self.cap = cv2.VideoCapture(rf"{self.filepath}")
            if not self.cap.isOpened():
                print("Error: Could not open video.")
                return
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.sleep_ms = int(np.round((1 / fps) * 1000))  
            ret, frame = self.cap.read()
            self.current_frame += 1  
            self.show_img(self.graphicsView,self.scene,frame)  
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)  
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  
            self.duration = self.frame_count / self.fps  
            self.Total_length.setText(self.transform_time(self.duration))  
            self.display_video_info()  
            self.btn_pause.show()

    def btn_change_pause(self):
        self.is_playing = not self.is_playing


    def forward(self):
        self.current_frame += int(100/6 * self.fps)  
        if self.current_frame >= self.frame_count:  
            self.current_frame = self.frame_count - 1  
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  
        self.Video_bar.setValue(int(self.current_frame/self.frame_count*100))

    def backward(self):
        self.current_frame -= int(50/6 * self.fps)  
        if self.current_frame < 0:  
            self.current_frame = 0  
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  
        self.Video_bar.setValue(int(self.current_frame/self.frame_count*100))

    def slider_moved(self, position):
        self.current_frame = int(position*self.frame_count/100)  
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  
        self.Current.setText(self.transform_time(self.duration))

    def item_clicked(self, item):
        transform_time = self.transform_time(item.text())  
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, transform_time*self.fps)  
        self.Video_bar.setValue(int(transform_time*self.fps/self.frame_count*100))

    def show_img(self, element, scene, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(element.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scene.clear()
        scene.addPixmap(scaled_pixmap)
        self.on_air_camera.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def transform_time(self, time_input):
        if isinstance(time_input, float):
            seconds = time_input
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        elif isinstance(time_input, str) and len(time_input.split(':')) == 3:
            time_parts = time_input.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
        else:
            raise ValueError("Invalid input format. Provide an integer or a 'HH:MM:SS' string.")
        
    def display_video_info(self):
        if self.cap is not None:  
            self.filename = os.path.basename(self.filepath)  
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  
            self.video_length = datetime.timedelta(seconds=int(self.length / self.fps))
            
            last_modified_timestamp = os.path.getmtime(self.filepath)
            last_modified_datetime = datetime.datetime.fromtimestamp(last_modified_timestamp)
            self.formatted_datetime = last_modified_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            info_text = f"파일명: {self.filename}\n\n프레임: {self.length}\n\n최근수정일자 : {self.formatted_datetime}\n\n크기: {self.width} x {self.height}\n\n영상 길이: {self.video_length}\n\nfps: {self.fps}"
            self.Video_info_text.setText(info_text)  

    def closeEvent(self, event):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                QMessageBox.Yes | QMessageBox.No)  
        if re == QMessageBox.Yes:  
            self.timer.stop()
            self.cap.release()
            event.accept()
            sys.exit()
        else:  
            event.ignore() 



