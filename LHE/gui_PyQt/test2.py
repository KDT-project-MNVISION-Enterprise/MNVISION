import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import numpy as np
from ultralytics import YOLO
import os

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/Video3.ui")[0] # Qt designer에서 작성한 ui 파일 불러오기
ort_session = YOLO('MNVISION/LHE/gui_PyQt/best.onnx') # 모델 파일 불러오기

class WindowClass(QMainWindow, form_class): # 
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # scene 등록하기 - 화면 출력 요소 지정.
        self.scene = QGraphicsScene() 
        self.graphicsView.setScene(self.scene) #graphicsView를 띄우는 코드
        
        # 시작버튼 지정
        self.btn_pause.clicked.connect(self.start_video) # 버튼 이름.clicked.connect : 버튼 활성화
        
        # 동영상 업로드 버튼
        self.Video_upload.clicked.connect(self.btn_fun_FileLoad)
        
        # 멈춤 버튼 연결
        self.btn_stop_start.clicked.connect(self.btn_change_pause)
        
        # 동영상 파일 경로 변수
        self.filepath=""
        # 동영상 저장 변수
        self.cap=None
        # 프레임 지정 시간 저장 변수
        self.sleep_ms=0
        # 재생/정지 플래그 변수
        self.is_playing=True

        self.Video_info_text=""
    
    # 동영상 불러오기
    def btn_fun_FileLoad(self):        
        fname=QFileDialog.getOpenFileName(self, 'Open file', './')        
        self.filepath=fname[0]
        print(self.filepath)
        self.cap = cv2.VideoCapture(rf"{self.filepath}") # 동영상 불러오기 
        fps = self.cap.get(cv2.CAP_PROP_FPS) # 시간 단위를 가지고 오는 
        self.sleep_ms = int(np.round((1 / fps) * 1000)) # 프레임 당 지연시간 측정 - 컴퓨터 환경에 구애받지 않고 같은 시간에 창이 뜨게 하기 위하여.
    
    # 비디오 시작
    def start_video(self):
        while True:
            if self.is_playing:
                ret, frame = self.cap.read() # 한 프레임 읽기
                
                # 객체 인식 수행
                results = ort_session(frame) # 모델 예측 수행

                # 결과 시각화
                frame = results[0].plot()

                # BGR -> RGB + QImage 생성
                qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888) # 중간다리
                pixmap = QPixmap(qt_image) # scene에 표시하기 위한 이미지로 

                # pixmap 스케일링 진행 
                scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.scene.clear() # 화면 요소 지웠다가,
                self.scene.addPixmap(scaled_pixmap) # 화면 출력하기
                self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio) # 항목이 뷰 내부에 꼭 맞게 들어가도록 한다
                
            key = cv2.waitKey(30)

        self.cap.release() # 코드가 끝나면 메모리 해제

    def btn_change_pause(self):
        self.is_playing = not self.is_playing

    def video_info(self) :
        self.cap = cv2.VideoCapture(rf"{self.filepath}")
        self.length=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps=int(self.cap.get(cv2.CAP_PROP_FPS))
        self.Video_info_text.setText(f"{self.length} 프레임, {self.width} x {self.height} 크기, {self.length/self.fps} fps")




    # 종료 확인
    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                    QMessageBox.Yes|QMessageBox.No) # 창 이름 : 종료 확인 / 메시지 : 종료하시겠습니까? / 버튼 : yes 또는 no

        if re == QMessageBox.Yes: # yes 선택 시
            QCloseEvent.accept() # 종료
        else: # no 선택 시
            QCloseEvent.ignore() # 종료 하지 않음.

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass() 
    myWindow.show()
    sys.exit(app.exec_())