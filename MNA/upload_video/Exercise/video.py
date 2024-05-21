import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.uic import loadUi
from media import CMultiMedia
import datetime
from VideoThread import VideoThread
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

class VideoApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(20, 20, 760, 540)

        self.video_thread = VideoThread(video_src=0)  # 0 for webcam or path to a video file
        self.video_thread.changePixmap.connect(self.set_image)

        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.setGeometry(20, 570, 75, 30)
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.setGeometry(100, 570, 75, 30)
        self.stop_button.clicked.connect(self.stop_video)

    def start_video(self):
        if not self.video_thread.isRunning():
            self.video_thread.start()

    def stop_video(self):
        self.video_thread.stop()

    @QtCore.pyqtSlot(QtGui.QImage)
    def set_image(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        resized_pixmap = pixmap.scaled(760, 540, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(resized_pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = VideoApp()
    main_window.show()
    sys.exit(app.exec_())
