import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtCore import QTimer

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle('Danger Detection System')
        self.setStyleSheet("background-color: red;")
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.blinkBackground)
        self.timer.start(500)  # 500ms마다 timeout 시그널 발생
        
        self.show()

    def blinkBackground(self):
        if self.palette().color(self.backgroundRole()) == 'red':
            self.setStyleSheet("background-color: white;")
        else:
            self.setStyleSheet("background-color: red;")

    def detectDanger(self):
        # 위험 감지 로직을 여기에 구현
        detected = True  # 위험을 감지했다고 가정

        if detected:
            QMessageBox.warning(self, 'Warning', 'Danger detected!', QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MyWidget()
    widget.detectDanger()
    sys.exit(app.exec_())