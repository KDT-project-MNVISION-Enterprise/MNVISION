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

form_class = uic.loadUiType("Video2/Video.ui")[0]
ort_session = YOLO('Streaming/best.onnx')

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        for i in range(1, 2):
            self.Log_text_2.addItem(f"위험상황 포착")
        self.Log_text_2.itemClicked.connect(self.dialog_open)
        self.model_flag=False
        self.btn_start_detection.clicked.connect(self.apply_model)
        self.cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()
        self.frame = None
        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.update_frame)  # 타이머가 만료될 때마다 update_frame 함수 호출
        self.timer.start(30)  # 30ms 간격으로 타이머 설정
        
        self.fps_list = []

    def apply_model(self):
        # btn_start_detection의 text를 "모델을 적용하는 중"이라고 바꾸기
        if self.btn_start_detection.text() == "탐지 시작":
            self.btn_start_detection.setText("모델 적용 중")
            self.model_flag = True
        else:
            self.btn_start_detection.setText("탐지 시작")
            self.model_flag = False

    def update_frame(self):
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap.release()
            return
        if self.model_flag:
            results = ort_session(self.frame)  # YOLO 객체 감지
            frame = results[0].plot()  # 결과 플롯
        self.show_img(self.frame)  # 이미지 표시
        self.save_frames(self.fps_list)
        
    def save_frames(self, fps_list):
        self.fps_list.append(self.frame.copy())
        if len(fps_list) > 90:
            del self.fps_list[0]

    def play_saved_frames(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        output_path = f'output_video_{timestamp}.mp4'
        frame_delay = 1 / 30  # 30 fps -> 1/30 seconds per frame

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
            QCoreApplication.processEvents()  # UI 갱신을 위해 이벤트 처리
            time.sleep(frame_delay)  # 1초에 30프레임 재생을 위해 딜레이 추가
            # Check if 6 seconds have passed
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
        scaled_pixmap = pixmap.scaled(self.play_frame_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene3.clear()
        self.scene3.addPixmap(scaled_pixmap)
        self.play_frame_view.fitInView(self.scene3.itemsBoundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)  # 종료 여부 확인 다이얼로그 표시
        if re == QMessageBox.Yes:  # Yes 버튼을 클릭한 경우
            self.timer.stop()
            self.cap.release()
            event.accept()
            sys.exit()
        else:  # No 버튼을 클릭한 경우
            event.ignore()  # 종료하지 않음.
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
