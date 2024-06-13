import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/practice/test.ui")[0]
ort_session = YOLO('MNVISION/LHE/gui_PyQt/practice/best.onnx')

class WindowClass2(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene2 = QGraphicsScene()
        self.camera.setScene(self.scene2)
        self.btn_detection.clicked.connect(self.start_detection_from_button)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.detection_flag = False

        self.danger_color = QColor(255, 0, 0, 128)
        self.normal_color = QColor(255, 255, 255, 0)
        self.current_color = self.normal_color

        self.current_frame = None
        self.current_detection_results = None

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.danger_detected = False  # danger_detected를 클래스의 속성으로 초기화

        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink)
        self.blink_timer.start(300)

    def blink(self):
        self.danger_detected = not self.danger_detected  # danger_detected 값을 토글

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap.release()
            return

        self.current_frame = frame

        if self.detection_flag:
            self.start_detection()
        else:
            self.show_img(frame)

    def start_detection_from_button(self):
        self.detection_flag = not self.detection_flag
        if self.detection_flag:
            self.btn_detection.setText("Stop detection")
        else:
            self.btn_detection.setText("Start detection")
            self.current_color = self.normal_color
            self.update()

    def start_detection(self):
        if self.current_frame is not None:
            results = ort_session(self.current_frame)
            self.current_detection_results = results
            self.show_img(self.current_frame, results)
            self.updateColor(results)

    def paintEvent(self, event):
        rect = QGraphicsRectItem()
        rect.setRect(0, 0, self.width, self.height)
        rect.setBrush(self.current_color)
        self.scene2.addItem(rect)

    def updateColor(self, results):
        # 탐지 결과를 이용하여 색상 업데이트
        if self.danger_detected:
            self.current_color = self.danger_color
        else:
            self.current_color = self.normal_color

        self.update()  # GUI 업데이트

    def show_img(self, frame, results=None):
        if results:
            frame = results[0].plot()

        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene2.clear()
        self.scene2.addPixmap(scaled_pixmap)
        self.camera.fitInView(self.scene2.itemsBoundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)
        if re == QMessageBox.Yes:
            QCloseEvent.accept()
            sys.exit()
        else:
            QCloseEvent.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass2() 
    myWindow.show()
    sys.exit(app.exec_())