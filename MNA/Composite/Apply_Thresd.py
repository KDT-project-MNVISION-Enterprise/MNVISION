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

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        for i in range(1, 2):
            self.Log_text_2.addItem(f"위험상황 포착")
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

    def start_video(self):
        self.btn_pause.hide()
        self.Current.show()
        self.Total_length.show()
        self.Video_bar.show()
        self.btn_forward.show()
        self.btn_prev.show()
        self.btn_stop_start.show()

        # Start video processing in a separate thread
        threading.Thread(target=self.process_video).start()

    def process_video(self):
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
                self.show_img(self.graphicsView,self.scene,frame)
            key = cv2.waitKey(30)

    def update_frame(self):
        ret, self.frame2 = self.cap2.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap2.release()
            return
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
            self.show_frame(fm)
            QCoreApplication.processEvents()
            time.sleep(frame_delay)
            if time.time() - start_time >= 6:
                break

        out.release()
        print(f"Video saved as {output_path}")

    def dialog_open(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle('3초 전 영상 재생')
        self.play_frame_view = QGraphicsView(self.dialog)
        self.scene3 = QGraphicsScene()
        self.play_frame_view.setScene(self.scene3)
        
                # 다이얼로그 레이아웃 설정
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(self.play_frame_view)
        self.dialog.setLayout(dialog_layout)

        self.message_label = QLabel("동영상 저장 중", self.dialog)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("QLabel {font-size: 24px; font-weight: bold; }")
        dialog_layout.addWidget(self.message_label)

        # 화면 크기의 1/4로 다이얼로그 크기 설정
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

    def show_frame(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.play_frame_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene3.clear()
        self.scene3.addPixmap(scaled_pixmap)
        self.play_frame_view.fitInView(self.scene3.itemsBoundingRect(), Qt.KeepAspectRatio)
        
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
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)
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

