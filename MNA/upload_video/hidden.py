import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 레이아웃 생성
        self.layout = QVBoxLayout()

        # 버튼 생성
        self.button = QPushButton('숨기기')
        self.button.clicked.connect(self.hide_button)

        # 보이기 버튼 생성
        self.show_button = QPushButton('다시 보이기')
        self.show_button.clicked.connect(self.show_hidden_button)

        # 레이아웃에 버튼 추가
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.show_button)

        # 레이아웃 설정
        self.setLayout(self.layout)

    def hide_button(self):
        # 버튼 숨기기
        self.button.hide()

    def show_hidden_button(self):
        # 숨겨진 버튼 다시 보이기
        self.button.show()

# QApplication 객체 생성
app = QApplication(sys.argv)

# MyWindow 객체 생성 및 표시
window = MyWindow()
window.show()

# 이벤트 루프 실행
sys.exit(app.exec_())
