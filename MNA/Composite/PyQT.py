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
        self.cap2 = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
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
            
            
        # 로그 텍스트 아이템 추가
        for i in range(1, 10):
            self.Log_text.addItem(f"00:0{i}:00")
        self.Log_text.itemClicked.connect(self.item_clicked)   
        
        self.frame2 = None # frame2 : 셀카, frame : 첫화면 
        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.update_frame)  # 타이머가 만료될 때마다 update_frame 함수 호출
        self.timer.start(30)  # 30ms 간격으로 타이머 설정

        # 변수 초기화
        self.filepath = ""  # 파일 경로
        self.cap = None  # 비디오 캡처 객체
        self.sleep_ms = 0  # 프레임 간격(ms)
        self.is_playing = True  # 재생 중 여부
        self.frame_count = None  # 총 프레임 수
        self.duration = None  # 비디오 총 재생 시간
        self.current_frame = 0  # 현재 프레임
        self.fps = None  # 프레임 속도
        self.progress_value = 0 # 프로그레스 바 값 
        self.flag=0 # 재생/정지 아이콘 변경 값
        self.fps_list = []
        
        
    def btn_fun_FileLoad(self):
        """
        파일을 로드하는 함수.

        파일 대화상자를 통해 비디오 파일을 선택하고 로드한다.

        :return: None
        """
        fname = QFileDialog.getOpenFileName(self, 'Open file', './','Video files (*.mp4 *.avi)')
        if fname[0]:
            self.filepath = fname[0]  # 파일 경로 저장
            self.cap = cv2.VideoCapture(rf"{self.filepath}")  # 비디오 캡처 객체 생성
            #self.cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
            if not self.cap.isOpened():
                print("Error: Could not open video.")
                return
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.sleep_ms = int(np.round((1 / fps) * 1000))  # 프레임 간격 계산
            ret, frame = self.cap.read()
            self.current_frame += 1  # 현재 프레임 증가
            self.show_img(self.graphicsView,self.scene,frame)  # 이미지 표시
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # FPS 설정
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수 설정
            self.duration = self.frame_count / self.fps  # 비디오 총 재생 시간 계산
            self.Total_length.setText(self.transform_time(self.duration))  # 총 재생 시간 텍스트 설정
            self.display_video_info()  # 비디오 정보 표시
            self.btn_pause.show()
            # 슬라이드 초기화 해주기!


    def transform_time(self, time_input):
        """
        시간 변환 함수.

        입력된 시간을 다른 형식으로 변환한다.

        :param time_input: 변환할 시간 (정수 또는 문자열)
        :type time_input: int or str
        :return: 변환된 시간
        :rtype: str or int
        :raises ValueError: 유효하지 않은 입력 형식일 경우 발생
        """
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
        """
        비디오 재생 함수.

        비디오를 재생한다.

        :return: None
        """
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(10)  # 45밀리초마다 update_progress 함수 호출
        self.btn_pause.hide()
        self.Current.show()
        self.Total_length.show()
        self.Video_bar.show()
        self.btn_forward.show()
        self.btn_prev.show()
        self.btn_stop_start.show()
        while True:
            if self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    self.is_playing = False
                    break
                self.current_frame += 1  # 현재 프레임 수 증가
                self.duration = self.current_frame / self.fps  # 재생 시간 업데이트
                self.Current.setText(self.transform_time(self.duration))  # 현재 재생 시간 업데이트
                results = ort_session(frame)  # YOLO 객체 감지
                frame = results[0].plot()  # 결과 플롯
                self.show_img(self.graphicsView,self.scene,frame)  # 이미지 표시
            key = cv2.waitKey(30)  # 키 입력 대기
        self.cap.release()  # 비디오 캡처 해제
        
    def update_progress(self):
        # 진행 상태 업데이트
        self.progress_value += 1
        self.progressBar.setValue(self.progress_value)

        # 100%에 도달하면 타이머 중지
        if self.progress_value >= 100:
            self.timer.stop()
            
            
            
            
            
            
    def apply_model(self):
        # btn_start_detection의 text를 "모델을 적용하는 중"이라고 바꾸기
        if self.btn_start_detection.text() == "탐지 시작":
            self.btn_start_detection.setText("모델 적용 중")
            self.model_flag = True
        else:
            self.btn_start_detection.setText("탐지 시작")
            self.model_flag = False

    def update_frame(self):
        ret, self.frame2 = self.cap2.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap2.release()
            return
        if self.model_flag:
            results = ort_session(self.frame2)  # YOLO 객체 감지
            self.frame2 = results[0].plot()  # 결과 플롯
        self.show_img(self.on_air_camera, self.scene2, self.frame2)  # 이미지 표시
        self.save_frames(self.fps_list)
        
    def save_frames(self, fps_list):
        self.fps_list.append(self.frame2.copy())
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
        """
        슬라이더 이동 이벤트 핸들러.

        슬라이더가 이동할 때 호출되는 함수.
        현재 프레임을 슬라이더 위치에 맞게 설정하고, 이에 따라 현재 시간을 업데이트한다.

        :param position: 슬라이더의 위치 (0~100 사이의 값)
        :type position: int
        :return: None
        """
        self.current_frame = int(position*self.frame_count/100)  # 슬라이더 위치에 맞는 프레임 계산
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  # 비디오 프레임 위치를 설정
        self.Current.setText(self.transform_time(self.duration))  # 현재 시간을 업데이트

    def item_clicked(self, item):
        """
        리스트 아이템 클릭 이벤트 핸들러.

        리스트 위젯의 항목이 클릭될 때 호출되는 함수.
        클릭된 항목의 텍스트를 초 단위로 변환하고 해당 프레임으로 이동한다.

        :param item: 클릭된 리스트 위젯의 아이템
        :type item: QListWidgetItem
        :return: None
        """

        transform_time = self.transform_time(item.text())  # 클릭된 항목의 시간을 초 단위로 변환
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, transform_time*self.fps)  # 비디오 프레임 위치를 설정
        #self.Video_bar 의 값을 바꾸기
        self.Video_bar.setValue(int(transform_time*self.fps/self.frame_count*100))  # 슬라이더 위치를 업데이트
    def btn_change_pause(self):
        """
        일시 정지 버튼 토글 함수.

        일시 정지 버튼의 상태를 변경하는 함수.
        """
        
        if not self.flag:
            icon = QIcon('Video/icon/play.png')  # 이미지 파일 경로를 설정
        else :
            icon = QIcon('Video/icon/stop-button.png')
        self.flag = not self.flag
        self.btn_stop_start.setIcon(icon)
        self.is_playing = not self.is_playing


    
    def forward(self):
        """
        영상을 앞으로 이동하는 함수.

        현재 프레임을 10초만큼 앞으로 이동시키고, 비디오의 마지막 프레임을 넘어가지 않도록 한다.

        :return: None
        """
        self.current_frame += int(100/6 * self.fps)  # 10초만큼 앞으로 이동
        if self.current_frame >= self.frame_count:  # 마지막 프레임을 넘어갔는지 확인
            self.current_frame = self.frame_count - 1  # 마지막 프레임으로 설정
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  # 비디오 프레임 위치를 설정
        self.Video_bar.setValue(int(self.current_frame/self.frame_count*100))

    def backward(self):
        """
        영상을 뒤로 이동하는 함수.

        현재 프레임을 10초만큼 뒤로 이동시키고, 첫 번째 프레임을 넘어가지 않도록 한다.

        :return: None
        """
        self.current_frame -= int(50/6 * self.fps)  # 10초만큼 뒤로 이동
        if self.current_frame < 0:  # 첫 번째 프레임을 넘어갔는지 확인
            self.current_frame = 0  # 첫 번째 프레임으로 설정
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  # 비디오 프레임 위치를 설정
        self.Video_bar.setValue(int(self.current_frame/self.frame_count*100))
    def closeEvent(self, event):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)  # 종료 여부 확인 다이얼로그 표시
        if re == QMessageBox.Yes:  # Yes 버튼을 클릭한 경우
            self.timer.stop()
            self.cap.release()
            self.cap2.release()
            event.accept()
            sys.exit()
        else:  # No 버튼을 클릭한 경우
            event.ignore()  # 종료하지 않음.
            
    def display_video_info(self):
        """
        비디오 정보를 화면에 표시하는 함수.

        비디오 파일의 정보를 읽어와서 화면에 표시한다.

        :return: None
        """
        if self.cap is not None:  # 비디오 캡처 객체가 있을 때만 실행
            self.filename = os.path.basename(self.filepath)  # 파일명 추출
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 프레임 수 추출
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비 추출
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이 추출
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # 프레임 속도 추출
            self.video_length = datetime.timedelta(seconds=int(self.length / self.fps))  # 비디오 길이 계산
            info_text = f"파일명: {self.filename}\n프레임: {self.length}\n크기: {self.width} x {self.height}\n영상 길이: {self.video_length}\nfps: {self.fps}"
            self.Video_info_text.setText(info_text)  # 비디오 정보를 화면에 표시
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
