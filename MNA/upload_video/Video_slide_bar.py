import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        # Create a widget for the window contents
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create a layout for the widgets
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        # Create the label to display the video frame
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Create the slider
        self.video_bar = QSlider(Qt.Horizontal, self)
        self.video_bar.setMinimum(0)
        self.video_bar.valueChanged.connect(self.slider_moved)
        self.layout.addWidget(self.video_bar)

        # Load the video file using OpenCV
        self.cap = cv2.VideoCapture('Long.avi')
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_bar.setMaximum(self.total_frames - 1)

        # Create a timer to update the video frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Adjust the timer interval as needed

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def slider_moved(self):
        frame_number = self.video_bar.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
