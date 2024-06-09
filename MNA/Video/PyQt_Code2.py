import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO

form_class = uic.loadUiType("Video/Video.ui")[0]
ort_session = YOLO('Streaming/best.onnx')

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)
        # btn_start_detection이 클릭되면 버튼_클릭 함수 실행
        self.model_flag=False
        self.btn_start_detection.clicked.connect(self.apply_model)
        self.cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.update_frame)  # 타이머가 만료될 때마다 update_frame 함수 호출
        self.timer.start(30)  # 30ms 간격으로 타이머 설정

    def apply_model(self):
        # btn_start_detection의 text를 "모델을 적용하는 중"이라고 바꾸기
        if self.btn_start_detection.text() == "모델을 적용 안하는 중!!!!":
            self.btn_start_detection.setText("모델 적용 중!!!!!!!!")
            self.model_flag = True
        else:
            self.btn_start_detection.setText("모델을 적용 안하는 중!!!!")
            self.model_flag = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap.release()
            return
        if self.model_flag:
            results = ort_session(frame)  # YOLO 객체 감지
            frame = results[0].plot()  # 결과 플롯
        self.show_img(frame)  # 이미지 표시

    def show_img(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.graphicsView_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene2.clear()
        self.scene2.addPixmap(scaled_pixmap)
        self.on_air_camera.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)  # 종료 여부 확인 다이얼로그 표시
        if re == QMessageBox.Yes:  # Yes 버튼을 클릭한 경우
            self.timer.stop()
            self.cap.release()
            event.accept()
            sys.exit()
        else:  # No 버튼을 클릭한 경우
            QCloseEvent.ignore()  # 종료하지 않음.
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
