import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO


form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/practice/test.ui")[0] # Qt designer에서 작성한 ui 파일 불러오기
ort_session = YOLO('MNVISION/LHE/gui_PyQt/practice/best.onnx') # 모델 파일 불러오기

class WindowClass2(QMainWindow, form_class): # c
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene2 = QGraphicsScene()
        self.camera.setScene(self.scene2)
        self.btn_detection.clicked.connect(self.start_detection_from_button)
        self.cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.update_frame)  # 타이머가 만료될 때마다 update_frame 함수 호출
        self.timer.start(30)  # 30ms 간격으로 타이머 설정

        self.detection_flag = False  # 탐지 플래그 초기화

        self.last_detection_time = QDateTime.currentDateTime()  # 마지막으로 탐지된 시간 기록

        self.no_detection_timer = QTimer(self)  # 타이머 생성
        self.no_detection_timer.timeout.connect(self.check_no_detection)  # 타임아웃 시 check_no_detection 메서드 호출
        self.no_detection_timer.start(500)  # 1초 간격으로 타이머 시작
        self.warning = False
        self.blink_flag = False  # 화면 깜박거리기용 플래그

        # CSS 파일을 로드하여 스타일을 적용
        self.loadStyleSheet("MNVISION/LHE/gui_PyQt/practice/style.css")

    def loadStyleSheet(self, filename):
        file = QFile(filename)
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        self.setStyleSheet(stream.readAll())


    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            self.timer.stop()
            self.cap.release()
            return

        if self.detection_flag:  # 탐지 플래그가 True이면 YOLO 모델 실행
            self.start_detection(frame)
        else:
            self.show_img(frame)

    def start_detection_from_button(self):
        self.detection_flag = not self.detection_flag  # 탐지 플래그를 True로 설정
        if self.detection_flag:
            self.btn_detection.setText("Stop detection")  # 버튼 텍스트 변경
        else:
            self.btn_detection.setText("Start detection")

    def start_detection(self, frame):
        results = ort_session(frame)  # YOLO 객체 감지
        self.timer2 = QTimer(self)
        class_labels_tensor = results[0].boxes.cls  # 클래스 라벨 텐서
        class_labels_list = class_labels_tensor.tolist()  # 텐서를 리스트로 변환

        for cls_index in class_labels_list:
            t = cls_index

            if t == 0 :
                self.warning = True
                if self.warning :
                    self.timer.singleShot(500, self.toggle_background_color)  # 500ms 후에 배경색 변경
            else :
                self.warning = False
                self.toggle_background_color()  # 배경색 초기화
        frame = results[0].plot()  # 결과 플롯
        self.show_img(frame)  # 이미지 표시

    def toggle_background_color(self):
        if self.warning:  # warning이 True일 때만 실행
            # 화면 깜박거리기용 플래그를 반전시킴
            self.blink_flag = not self.blink_flag

            if self.blink_flag:
                # 빨강색과 투명한 사각형을 번갈아가며 화면에 표시하여 깜박거리는 효과를 만듦
                red_transparent_rect = QImage(self.camera.width(), self.camera.height(), QImage.Format_ARGB32)
                red_transparent_rect.fill(Qt.transparent)
                painter = QPainter(red_transparent_rect)
                painter.fillRect(red_transparent_rect.rect(), QColor(255, 0, 0, 30))
                painter.end()

                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(red_transparent_rect))
                self.scene2.clear()
                self.scene2.addItem(pixmap_item)
                self.camera.fitInView(self.scene2.itemsBoundingRect(), Qt.KeepAspectRatio)
            else:
                self.scene2.clear()  # 이전에 추가된 사각형 제거하여 투명한 화면으로 전환



    def show_img(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene2.clear()
        self.scene2.addPixmap(scaled_pixmap)
        self.camera.fitInView(self.scene2.itemsBoundingRect(), Qt.KeepAspectRatio)


    def check_no_detection(self):
        current_time = QDateTime.currentDateTime()  # 현재 시간
        time_difference = self.last_detection_time.secsTo(current_time)  # 마지막 탐지 시간과 현재 시간의 차이 (초)

        if time_difference > 3:  # 3초 이상 경과한 경우
            self.toggle_background_color()  # 배경색 초기화

    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                  QMessageBox.Yes | QMessageBox.No)  # 종료 여부 확인 다이얼로그 표시
        if re == QMessageBox.Yes:  # Yes 버튼을 클릭한 경우
            QCloseEvent.accept()  # 종료
            sys.exit()
        else:  # No 버튼을 클릭한 경우
            QCloseEvent.ignore()  # 종료하지 않음.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass2() 
    myWindow.show()
    sys.exit(app.exec_())