from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsView, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic

form_class = uic.loadUiType("Composite/Video.ui")[0]

class VideoView(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        self.btn_start_detection.clicked.connect(self.toggle_model_detection)
        self.Video_upload.clicked.connect(self.open_file_dialog)
        self.btn_pause.clicked.connect(self.start_video)
        self.btn_stop_start.clicked.connect(self.toggle_video_playback)
        self.btn_forward.clicked.connect(self.forward_video)
        self.btn_prev.clicked.connect(self.backward_video)
        self.Video_bar.valueChanged.connect(self.seek_video)

        self.hide_video_controls()

    def hide_video_controls(self):
        self.btn_pause.hide()
        self.Current.hide()
        self.Total_length.hide()
        self.Video_bar.hide()
        self.btn_forward.hide()
        self.btn_prev.hide()
        self.btn_stop_start.hide()

    def show_video_controls(self):
        self.btn_pause.show()
        self.Current.show()
        self.Total_length.show()
        self.Video_bar.show()
        self.btn_forward.show()
        self.btn_prev.show()
        self.btn_stop_start.show()

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', './', 'Video files (*.mp4 *.avi)')
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        raise NotImplementedError("This method should be implemented in the Controller class.")

    def start_video(self):
        raise NotImplementedError("This method should be implemented in the Controller class.")

    def toggle_model_detection(self):
        raise NotImplementedError("This method should be implemented in the Controller class.")

    def update_frame(self, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def toggle_video_playback(self):
        raise NotImplementedError("This method should be implemented in the Controller class.")

    def forward_video(self):
        raise NotImplementedError("This method should be implemented in the Controller class.")

    def backward_video(self):
        raise NotImplementedError("This method should be implemented in the Controller class.")

    def seek_video(self, position):
        raise NotImplementedError("This method should be implemented in the Controller class.")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()