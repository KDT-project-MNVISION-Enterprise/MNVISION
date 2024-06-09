import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QTextBrowser, QVBoxLayout, QWidget, QFileDialog, QDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QTimer

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.current_frame = None
        self.setMouseTracking(True)
        self.drawing = False

    def setFrame(self, frame):
        self.current_frame = frame
        self.update()

    def getFrameSize(self):
        if self.current_frame is not None:
            return self.current_frame.shape[1], self.current_frame.shape[0]  # (width, height)
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and len(self.points) < 4:
            self.points.append(event.pos())
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_frame is not None:
            image = QImage(self.current_frame.data, self.current_frame.shape[1], self.current_frame.shape[0], self.current_frame.strides[0], QImage.Format_RGB888)
            painter = QPainter(self)
            painter.drawImage(0, 0, image)
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            for point in self.points:
                painter.drawEllipse(point, 5, 5)
            if len(self.points) == 4:
                sorted_points = self.sortPoints(self.points)
                pen.setColor(Qt.blue)
                painter.setPen(pen)
                painter.drawPolygon(*sorted_points)

    def sortPoints(self, points):
        points = sorted(points, key=lambda p: (p.x(), p.y()))
        top_points = sorted(points[:2], key=lambda p: p.y())
        bottom_points = sorted(points[2:], key=lambda p: p.y())
        return top_points[0], top_points[1], bottom_points[1], bottom_points[0]

class SelectAreaDialog(QDialog):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Area')
        self.video_widget = VideoWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_widget)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(30)

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_widget.setFrame(frame)
            if not self.isVisible():  # If the dialog is not visible yet, set its size
                frame_size = self.video_widget.getFrameSize()
                if frame_size:
                    self.resize(frame_size[0] + 100, frame_size[1] + 100)  # Add extra space
                self.show()
        else:
            self.timer.stop()

    def mouseReleaseEvent(self, event):
        if len(self.video_widget.points) == 4:
            sorted_points = self.video_widget.sortPoints(self.video_widget.points)
            coordinates = [(p.x(), p.y()) for p in sorted_points]
            self.text_browser.setText(str(coordinates))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt Video Point Selector')

        self.text_browser = QTextBrowser()
        self.button_load_video = QPushButton('Load Video')
        self.button_select_area = QPushButton('영역 지정하기')
        self.button_load_video.clicked.connect(self.load_video)
        self.button_select_area.clicked.connect(self.open_select_area_dialog)

        layout = QVBoxLayout()
        layout.addWidget(self.text_browser)
        layout.addWidget(self.button_load_video)
        layout.addWidget(self.button_select_area)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.video_path = None

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if self.video_path:
            self.text_browser.setText(f"Loaded video: {self.video_path}")

    def open_select_area_dialog(self):
        if self.video_path:
            dialog = SelectAreaDialog(self.video_path, self)
            dialog.exec_()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
