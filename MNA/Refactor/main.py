import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from VideoController import VideoController
from CameraController import CameraController
from LogController import LogController
from PyQt5.QtWidgets import QMessageBox

form_class = uic.loadUiType("Composite/Video.ui")[0]

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 다른 클래스 인스턴스 생성 및 초기화
        self.video_controller = VideoController(self)
        self.camera_controller = CameraController(self)
        self.log_controller = LogController(self)
        
        # 시그널-슬롯 연결
        self.Video_upload.clicked.connect(self.video_controller.open_file_dialog)
        self.btn_pause.clicked.connect(self.video_controller.start_video)
        self.btn_stop_start.clicked.connect(self.video_controller.toggle_video_playback)
        self.btn_forward.clicked.connect(self.video_controller.forward_video)
        self.btn_prev.clicked.connect(self.video_controller.backward_video)
        self.Video_bar.valueChanged.connect(self.video_controller.seek_video)
        self.btn_start_detection.clicked.connect(self.camera_controller.toggle_model_detection)
        self.Log_text_2.itemClicked.connect(self.log_controller.show_dialog)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.video_controller.release_resources()
            self.camera_controller.release_resources()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())