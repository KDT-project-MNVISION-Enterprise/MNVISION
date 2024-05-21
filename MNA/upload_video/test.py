import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import numpy as np
from ultralytics import YOLO

form_class = uic.loadUiType("MNA/upload_video/Video.ui")[0]
ort_session = YOLO('MNA/Streaming/best.onnx')

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # scene 등록하기
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        
        # 시작버튼 지정
        self.btn_stop_start.clicked.connect(self.start_video)

    def start_video(self):
        cap = cv2.VideoCapture("MNA/upload_video/person.avi") # 동영상 불러오기 
        fps = cap.get(cv2.CAP_PROP_FPS)
        sleep_ms = int(np.round((1 / fps) * 1000))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 객체 인식 수행
            results = ort_session(frame)

            # 결과 시각화
            frame = results[0].plot()
            # BGR -> RGB + QImage 생성
            qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap(qt_image) # scene에 표시하기 위한 이미지로 

            # pixmap 스케일링 진행 
            scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.scene.clear() # 화면 요소 지웠다가,
            self.scene.addPixmap(scaled_pixmap) # 화면 출력하기
            self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            if cv2.waitKey(sleep_ms) & 0xFF == ord('q'):
                break

        cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass() 
    myWindow.show()
    sys.exit(app.exec_())
