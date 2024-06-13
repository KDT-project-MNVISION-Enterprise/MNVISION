import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import numpy as np
from ultralytics import YOLO

form_class = uic.loadUiType("MNVISION/MNA/upload_video/Video.ui")[0] # Qt designer에서 작성한 ui 파일 불러오기
ort_session = YOLO('MNVISION/MNA/Streaming/best.onnx') # 모델 파일 불러오기

class WindowClass(QMainWindow, form_class): # 
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # scene 등록하기 - 화면 출력 요소 지정.
        self.scene = QGraphicsScene() 
        self.graphicsView.setScene(self.scene) #graphicsView를 띄우는 코드
        
        # 시작버튼 지정
        self.btn_stop_start.clicked.connect(self.start_video) # 버튼 이름.clicked.connect : 버튼 활성화

    def start_video(self):
        cap = cv2.VideoCapture("MNVISION/MNA/upload_video/person.avi") # 동영상 불러오기 
        fps = cap.get(cv2.CAP_PROP_FPS) # 시간 단위를 가지고 오는 
        sleep_ms = int(np.round((1 / fps) * 1000)) # 프레임 당 지연시간 측정 - 컴퓨터 환경에 구애받지 않고 같은 시간에 창이 뜨게 하기 위하여.

        while True:
            ret, frame = cap.read() # 한 프레임 읽기
            if not ret: #결과물 x
                break
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

            if cv2.waitKey(sleep_ms) & 0xFF == ord('q'): # 키 입력을 기다리는 함수 -> q를 누르면 정지.
                break

        cap.release() # 코드가 끝나면 메모리 해제

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass() 
    myWindow.show()
    sys.exit(app.exec_())
