import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtCore import Qt, QTimer

class DangerDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 800, 600)  # 창 크기 설정
        self.setWindowTitle("Danger Detection GUI")

        self.danger_color = QColor(255, 0, 0, 128)  # 빨간색 투명도 설정
        self.normal_color = QColor(255, 255, 255, 0)  # 투명 색상 설정
        self.current_color = self.normal_color  # 현재 색상 설정

        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.updateColor)  # 타이머 이벤트에 updateColor 함수 연결
        self.timer.start(500)  # 1초 주기로 타이머 시작

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QBrush(self.current_color))  # 현재 색상으로 사각형 그리기
        painter.drawRect(self.rect())

    def updateColor(self):
        if self.current_color == self.normal_color:
            self.current_color = self.danger_color  # 위험 상황 시 빨간색으로 변경
        else:
            self.current_color = self.normal_color  # 정상 상황 시 투명색으로 변경

        self.update()  # GUI 업데이트

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DangerDetectionGUI()
    window.show()
    sys.exit(app.exec_())