import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar
from PyQt5.QtCore import QTimer

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 레이아웃 생성
        self.layout = QVBoxLayout()

        # QProgressBar 생성
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)  # 최대값 설정
        self.layout.addWidget(self.progress_bar)

        # 레이아웃 설정
        self.setLayout(self.layout)

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(45)  # 40밀리초마다 update_progress 함수 호출

        # 진행 상태 초기화
        self.progress_value = 0

    def update_progress(self):
        # 진행 상태 업데이트
        self.progress_value += 1
        self.progress_bar.setValue(self.progress_value)

        # 100%에 도달하면 타이머 중지
        if self.progress_value >= 100:
            self.timer.stop()

# QApplication 객체 생성
app = QApplication(sys.argv)

# MyWindow 객체 생성 및 표시
window = MyWindow()
window.show()

# 이벤트 루프 실행
sys.exit(app.exec_())
