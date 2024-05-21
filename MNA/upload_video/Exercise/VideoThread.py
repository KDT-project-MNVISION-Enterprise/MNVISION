import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

class VideoThread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, video_src):
        super().__init__()
        self.video_src = "./person.avi"
        self.running = False
        

    def run(self):
        cap = cv2.VideoCapture(self.video_src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sleep_ms = int(np.round((1 / fps) * 1000))  # Adjusting sleep time based on fps

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.changePixmap.emit(qt_image)

            if cv2.waitKey(sleep_ms) & 0xFF == ord('q'):
                break

        cap.release()

    def stop(self):
        self.running = False
        self.wait()