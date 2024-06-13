from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.view = QGraphicsView()
        self.view.setMinimumSize(600, 400)  # 가로 600, 세로 400 크기로 설정
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        self.button = QPushButton("Toggle Warning")
        self.button.clicked.connect(self.toggle_warning)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.button)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.toggle_background_color)

        self.warning = False

    def toggle_warning(self):
        self.warning = not self.warning
        if self.warning:
            self.button.setText("Turn Off Warning")
            self.timer.start(500)  # 500 milliseconds
        else:
            self.button.setText("Turn On Warning")
            self.timer.stop()
            self.view.setStyleSheet("")  # Reset stylesheet

    def toggle_background_color(self):
        current_stylesheet = self.view.styleSheet()
        if "background-color: rgba(255, 0, 0, 100);" in current_stylesheet:
            self.view.setStyleSheet("")
        else:
            self.view.setStyleSheet("background-color: rgba(255, 0, 0, 100);")

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()