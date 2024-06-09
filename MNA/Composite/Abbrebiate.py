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

form_class = uic.loadUiType("Composite/Video.ui")[0]
ort_session = YOLO('Composite/best.onnx')
ort_session2 = YOLO('Composite/best.onnx')

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        for i in range(1, 2):
            self.Log_text_2.addItem(f"위험상황 포착") # 수정 필요
        self.Log_text_2.itemClicked.connect(self.dialog_open)
        self.model_flag=False
        self.btn_start_detection.clicked.connect(self.apply_model)
        self.cap2 = cv2.VideoCapture(0)
        if not self.cap2.isOpened():
            print("Error: Could not open video.")
            sys.exit()

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

        self.frame2 = None 
        self.timer = QTimer(self) 
        self.timer.timeout.connect(self.update_frame) 
        self.timer.start(30)  

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

    def transform_time(self, time_input):
        if isinstance(time_input, float):
            return f"{int(time_input // 3600):02}:{int((time_input % 3600) // 60):02}:{int(time_input % 60):02}"
        elif isinstance(time_input, str) and len(time_input.split(':')) == 3:
            hours, minutes, seconds = map(int, time_input.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError("Invalid input format. Provide an integer or a 'HH:MM:SS' string.")

    def start_video(self):
        self.timer = QTimer()
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
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                self.timer_video.stop()  # 비디오 끝에 도달하면 타이머 중지
                self.cap.release()
                return
            self.current_frame += 2  # 현재 프레임 수  2증가
            self.duration = self.current_frame / self.fps  # 재생 시간 업데이트
            self.Current.setText(self.transform_time(self.duration))  # 현재 재생 시간 업데이트
            results = ort_session2(frame)  # YOLO 객체 감지
            frame = results[0].plot()  # 결과 플롯
            self.show_img(self.graphicsView, self.scene, frame)  # 이미지 표시


    def update_progress(self):
        self.progress_value += 1
        self.progressBar.setValue(self.progress_value)
        if self.progress_value >= 100:
            self.timer.stop()

    def apply_model(self):
        self.model_flag = not self.model_flag
        self.btn_start_detection.setText("모델 적용 중" if self.model_flag else "탐지 시작")


    def update_frame(self):
        ret, self.frame2 = self.cap2.read()
        if self.model_flag:
            results = ort_session(self.frame2)  
            self.frame2 = results[0].plot()  
        self.show_img(self.on_air_camera, self.scene2, self.frame2)  
        self.save_frames(self.fps_list)

    def save_frames(self, fps_list):
        self.fps_list.append(self.frame2.copy())
        if len(fps_list) > 90:
            del self.fps_list[0]

    def play_saved_frames(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        output_path = f'output_video_{timestamp}.mp4'
        frame_delay = 1 / 30  
        if len(self.fps_list) == 0:
            print("Error: No frames to save.")
            return
        frame_height, frame_width, _ = self.fps_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
        start_time = time.time()
        for fm in self.fps_list:
            out.write(fm)
            self.show_img(self.play_frame_view, self.scene3,fm)
            QCoreApplication.processEvents()  
            time.sleep(frame_delay)  
            if time.time() - start_time >= 6:
                break
        out.release()

    def dialog_open(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle('3초 전 영상 재생')
        self.play_frame_view = QGraphicsView(self.dialog)
        self.scene3 = QGraphicsScene()
        self.play_frame_view.setScene(self.scene3)
        
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(self.play_frame_view)
        self.dialog.setLayout(dialog_layout)

        self.message_label = QLabel("동영상 저장 중", self.dialog)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("QLabel {font-size: 24px; font-weight: bold; }")
        dialog_layout.addWidget(self.message_label)

        screen = QDesktopWidget().screenGeometry()
        width = screen.width() // 2
        height = screen.height() // 2
        self.dialog.resize(width, height)

        self.dialog.show()
        self.play_saved_frames()
        self.message_label.setText("동영상 저장 완료")
        
    def show_img(self, element, scene, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(element.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scene.clear()
        scene.addPixmap(scaled_pixmap)
        self.on_air_camera.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
    def slider_moved(self, position):
        self.current_frame = int(position*self.frame_count/100)  
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  
        self.Current.setText(self.transform_time(self.duration))  

    def item_clicked(self, item):
        transform_time = self.transform_time(item.text())  
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, transform_time*self.fps)  
        self.Video_bar.setValue(int(transform_time*self.fps/self.frame_count*100))  

    def btn_change_pause(self):
        if not self.flag:
            icon = QIcon('Video/icon/play.png')  
        else :
            icon = QIcon('Video/icon/stop-button.png')
        self.flag = not self.flag
        self.btn_stop_start.setIcon(icon)
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
        
    def closeEvent(self, event):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?", QMessageBox.Yes | QMessageBox.No)  
        if re == QMessageBox.Yes:  
            self.timer.stop()
            self.cap.release()
            self.cap2.release()
            event.accept()
            sys.exit()
        else:  
            event.ignore()  
            
    def display_video_info(self):
        if self.cap is not None:  
            self.filename = os.path.basename(self.filepath)  
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  
            self.video_length = datetime.timedelta(seconds=int(self.length / self.fps))  
            info_text = f"파일명: {self.filename}\n프레임: {self.length}\n크기: {self.width} x {self.height}\n영상 길이: {self.video_length}\nfps: {self.fps}"
            self.Video_info_text.setText(info_text)  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())