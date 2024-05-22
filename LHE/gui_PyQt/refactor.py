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

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/Video.ui")[0] # Qt designer에서 작성한 ui 파일 불러오기
ort_session = YOLO('MNVISION/LHE/gui_PyQt/best.onnx') # 모델 파일 불러오기

class WindowClass(QMainWindow, form_class): # 
    def __init__(self):
        """
        클래스 초기화 함수.

        QMainWindow와 form_class를 상속받아 윈도우 클래스를 초기화한다.

        :return: None
        """
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

        # 로그 텍스트 아이템 추가
        for i in range(1, 10):
            self.Log_text.addItem(f"00:0{i}:00")
        self.Log_text.itemClicked.connect(self.item_clicked)

        # 변수 초기화
        self.filepath = ""  # 파일 경로
        self.cap = None  # 비디오 캡처 객체
        self.sleep_ms = 0  # 프레임 간격(ms)
        self.is_playing = False  # 재생 중 여부
        self.frame_count = None  # 총 프레임 수
        self.duration = None  # 비디오 총 재생 시간
        self.current_frame = 0  # 현재 프레임
        self.fps = None  # 프레임 속도

    def btn_fun_FileLoad(self):
        """
        파일을 로드하는 함수.

        파일 대화상자를 통해 비디오 파일을 선택하고 로드한다.

        :return: None
        """
        fname = QFileDialog.getOpenFileName(self, 'Open file', './', "Video files (*.mp4 *.avi)")
        self.filepath = fname[0]  # 파일 경로 저장
        self.cap = cv2.VideoCapture(rf"{self.filepath}")  # 비디오 캡처 객체 생성
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.sleep_ms = int(np.round((1 / fps) * 1000))  # 프레임 간격 계산
        ret, frame = self.cap.read()
        self.current_frame += 1  # 현재 프레임 증가
        self.show_img(frame)  # 이미지 표시
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # FPS 설정
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수 설정
        self.duration = self.frame_count / self.fps  # 비디오 총 재생 시간 계산
        self.Total_length.setText(self.transform_time(self.duration))  # 총 재생 시간 텍스트 설정
        self.display_video_info()  # 비디오 정보 표시


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
                self.show_img(frame)  # 이미지 표시
            key = cv2.waitKey(30)  # 키 입력 대기
        self.cap.release()  # 비디오 캡처 해제

    
    def show_img(self, frame):
        """
        이미지 표시 함수.

        주어진 프레임을 QGraphicsView에 표시한다.

        :param frame: 표시할 이미지 프레임
        :type frame: numpy.ndarray
        :return: None
        """
        # OpenCV 프레임을 Qt 이미지로 변환
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        # Qt 이미지를 QPixmap으로 변환
        pixmap = QPixmap(qt_image)
        # 이미지 크기를 QGraphicsView에 맞게 조절하여 유지하면서 스무딩 적용
        scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # scene을 초기화하고 scaled_pixmap을 추가하여 이미지 표시
        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)
        # graphicsView가 scene의 경계에 맞게 자동 조정되도록 설정
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)


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

    def btn_change_pause(self):
        """
        일시 정지 버튼 토글 함수.

        일시 정지 버튼의 상태를 변경하는 함수.
        """
        self.is_playing = not self.is_playing  # 일시 정지 상태를 토글


    
    def forward(self):
        """
        영상을 앞으로 이동하는 함수.

        현재 프레임을 10초만큼 앞으로 이동시키고, 비디오의 마지막 프레임을 넘어가지 않도록 한다.

        :return: None
        """
        self.current_frame += int(10 * self.fps)  # 10초만큼 앞으로 이동
        if self.current_frame >= self.frame_count:  # 마지막 프레임을 넘어갔는지 확인
            self.current_frame = self.frame_count - 1  # 마지막 프레임으로 설정
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  # 비디오 프레임 위치를 설정

    def backward(self):
        """
        영상을 뒤로 이동하는 함수.

        현재 프레임을 10초만큼 뒤로 이동시키고, 첫 번째 프레임을 넘어가지 않도록 한다.

        :return: None
        """
        self.current_frame -= int(10 * self.fps)  # 10초만큼 뒤로 이동
        if self.current_frame < 0:  # 첫 번째 프레임을 넘어갔는지 확인
            self.current_frame = 0  # 첫 번째 프레임으로 설정
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)  # 비디오 프레임 위치를 설정

    
    
    def closeEvent(self, QCloseEvent):
        """
        창을 닫을 때의 이벤트 핸들러 함수.

        사용자에게 종료 여부를 확인하고, Yes 버튼을 클릭하면 종료하고, No 버튼을 클릭하면 종료하지 않는다.

        :param QCloseEvent: 닫기 이벤트 객체
        :return: None
        """
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)  # 종료 여부 확인 다이얼로그 표시
        if re == QMessageBox.Yes:  # Yes 버튼을 클릭한 경우
            QCloseEvent.accept()  # 종료
            self.is_playing=False
        else:  # No 버튼을 클릭한 경우
            QCloseEvent.ignore()  # 종료하지 않음.


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
