import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import time
import os
import re

form_class = uic.loadUiType("MNVISION/LHE/gui_PyQt/OOP(Object Oriented Programming)/Video.ui")[0]
ort_session = YOLO('MNVISION/LHE/gui_PyQt/OOP(Object Oriented Programming)/best.onnx')
ort_session2 = YOLO('MNVISION/LHE/gui_PyQt/OOP(Object Oriented Programming)/best.onnx')

class VideoProcessor:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.duration = None
        self.current_frame = 0
        self.is_playing = True

        if filepath:
            self.load_video(filepath)

    def load_video(self, filepath):
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        return self.duration

    def get_frame(self):
        if self.cap is not None and self.is_playing:
            ret, frame = self.cap.read()
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 2
                return frame
            else:
                self.is_playing = False

    def set_frame_position(self, frame_number):
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def release(self):
        if self.cap is not None:
            self.cap.release()

class CameraProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open camera.")
        self.frame = None
        self.model_flag = False

    def get_frame(self):
        ret, self.frame = self.cap.read()
        return self.frame if ret else None

    def apply_model(self, frame, model):
        results = model(frame)
        return results[0].plot()

    def release(self):
        if self.cap is not None:
            self.cap.release()

class FrameSaver:
    def __init__(self):
        self.frames = []
        self.comboBox.clicked.connect(self.save_frame)

    def save_frame(self, frame):

        self.frames.append(frame.copy())
        if len(self.frames) > 90:
            del self.frames[0]

    def save_to_video(self, output_path, fps=15):
        if len(self.frames) == 0:
            raise ValueError("Error: No frames to save.")
        frame_height, frame_width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        start_time = time.time()
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        for frame in self.frames:
            out.write(frame)
            if time.time() - start_time >= 6:
                    break
        out.release()

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
    def __init__(self, video_path=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Area')
        self.video_widget = VideoWidget()
        self.setWindowState(Qt.WindowMaximized)
        self.coordinates = None
        
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
            if not self.isVisible():  
                frame_size = self.video_widget.getFrameSize()
                if frame_size:
                    self.resize(frame_size[0] + 100, frame_size[1] + 100)  # Add extra space
                self.show()
        else:
            self.timer.stop()

    def mouseReleaseEvent(self, event):
        if len(self.video_widget.points) == 4:
            sorted_points = self.video_widget.sortPoints(self.video_widget.points)
            self.coordinates = [(p.x(), p.y()) for p in sorted_points]
            self.accept()  # Close the dialog and emit accepted signal
        else:
            super().mouseReleaseEvent(event)  # Call the base class implementation

    def closeEvent(self, event):
        self.reject()  # Emit the rejected signal when the dialog is closed

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.on_air_camera.setScene(self.scene2)

        self.video_processor = VideoProcessor()
        self.camera_processor = CameraProcessor()
        self.frame_saver = FrameSaver()
        self.video_widget = VideoWidget()

        self.model_flag = False
        self.rectangle1_flag = False
        self.rectangle2_flag = False
        self.btn_start_detection.clicked.connect(self.toggle_model)
        self.btn_pause.clicked.connect(self.start_video)
        self.btn_pause.hide()
        self.Current.hide()
        self.Total_length.hide()
        self.Video_bar.hide()
        self.btn_forward.hide()
        self.btn_prev.hide()
        self.btn_stop_start.hide()
        self.Video_upload.clicked.connect(self.load_video_file)
        self.btn_stop_start.clicked.connect(self.toggle_play_pause)
        self.btn_forward.clicked.connect(self.forward)
        self.btn_prev.clicked.connect(self.backward)
        self.Video_bar.valueChanged.connect(self.slider_moved)
        self.progressBar.setValue(0)
        self.rack_btn_1.clicked.connect(lambda: self.open_select_area_dialog(1))
        self.rack_btn_2.clicked.connect(lambda: self.open_select_area_dialog(2))
        self.rack_btn_3.clicked.connect(lambda: self.draw_rectangle(1))
        self.rack_btn_4.clicked.connect(lambda: self.draw_rectangle(2))
        
        self.checkBox2.clicked.connect(self.muting)
        
        for i in range(1, 10):
            self.Log_text.addItem(f"00:0{i}:00")
        self.Log_text.itemClicked.connect(self.item_clicked)
        self.Log_text_2.addItem("위험 감지!")
        self.Log_text_2.itemClicked.connect(self.dialog_open)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.timer_video = QTimer(self)
        self.timer_video.timeout.connect(self.process_video)
        self.timer.start(15)
        
    def draw_rectangle(self,num):
        if num==1:
            text = self.rack_text_1.toPlainText()
            pattern = r'\((\d+), (\d+)\)'
            matches = re.findall(pattern, text)
            self.points1 = [(int(x), int(y)) for x, y in matches]
            self.rectangle1_flag = not self.rectangle1_flag
        if num==2:
            text = self.rack_text_2.toPlainText()
            pattern = r'\((\d+), (\d+)\)'
            matches = re.findall(pattern, text)
            self.points2 = [(int(x), int(y)) for x, y in matches]
            self.rectangle2_flag = not self.rectangle2_flag
        
    def open_select_area_dialog(self, num):
        if num==1:
            self.rectangle1_flag = False
        if num==2:
            self.rectangle2_flag = False
        self.coordinate_box=num
        self.camera_processor.cap.release() # 메모리 해제 
        self.dialog = SelectAreaDialog(0, self)
        self.dialog.finished.connect(self.on_dialog_finished)
        self.dialog.show()

    def on_dialog_finished(self, result):
        if result == QDialog.Rejected:
            print("Dialog was closed")
        print(str(self.dialog.coordinates))
        if self.coordinate_box == 1 and self.dialog.coordinates :
            self.rack_text_1.setText(str(self.dialog.coordinates))
        elif self.coordinate_box == 2 and self.dialog.coordinates:
            self.rack_text_2.setText(str(self.dialog.coordinates))
        else:
            print("Invalid coordinate box number")
        self.dialog.deleteLater()  # 다이얼로그 객체 삭제
        self.camera_processor.cap = cv2.VideoCapture(0)
        
            
    def load_video_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './', 'Video files (*.mp4 *.avi)')
        if fname[0]:
            duration = self.video_processor.load_video(fname[0])
            self.Total_length.setText(self.transform_time(duration))  
            self.display_first_frame()
            self.display_video_info()
            self.btn_pause.show()

    def display_first_frame(self):
        frame = self.video_processor.get_frame()
        if frame is not None:
            self.show_img(self.graphicsView, self.scene, frame)

    def display_video_info(self):
        info_text = (f"파일명: {os.path.basename(self.video_processor.filepath)}\n"
                     f"프레임: {self.video_processor.frame_count}\n"
                     f"크기: {self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n"
                     f"영상 길이: {str(datetime.timedelta(seconds=int(self.video_processor.duration)))}\n"
                     f"fps: {self.video_processor.fps}")
        self.Video_info_text.setText(info_text)

    def transform_time(self, time_input):
        if isinstance(time_input, float):
            return f"{int(time_input // 3600):02}:{int((time_input % 3600) // 60):02}:{int(time_input % 60):02}"
        elif isinstance(time_input, str) and len(time_input.split(':')) == 3:
            hours, minutes, seconds = map(int, time_input.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError("Invalid input format. Provide an integer or a 'HH:MM:SS' string.")

    def start_video(self):
        self.timer_video.start(30)
        self.btn_pause.hide()
        self.Current.show()
        self.Total_length.show()
        self.Video_bar.show()
        self.btn_forward.show()
        self.btn_prev.show()
        self.btn_stop_start.show()

    def process_video(self):
        frame = self.video_processor.get_frame()
        if self.video_processor.is_playing: # 동영상 모델 적용 상시
            frame = self.camera_processor.apply_model(frame, ort_session2)
            self.show_img(self.graphicsView, self.scene, frame)
            self.update_current_time()

    def toggle_model(self):
        self.model_flag = not self.model_flag
        self.btn_start_detection.setText("모델 적용 중" if self.model_flag else "탐지 시작")

    def update_frame(self):
        frame = self.camera_processor.get_frame()
        if self.rectangle1_flag:
            points_int = np.array(self.points1, dtype=np.int32)
            cv2.polylines(frame, [points_int], True, (255, 0, 0), thickness=2)
        if self.rectangle2_flag:
            points_int = np.array(self.points2, dtype=np.int32)
            cv2.polylines(frame, [points_int], True, (255, 0, 0), thickness=2)
        if frame is not None:
            if self.model_flag:
                frame = self.camera_processor.apply_model(frame, ort_session)
            self.show_img(self.on_air_camera, self.scene2, frame)
            self.frame_saver.save_frame(frame)
            self.video_widget.setFrame(frame)

    def show_img(self, element, scene, frame):
        qt_image = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        scaled_pixmap = pixmap.scaled(element.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scene.clear()
        scene.addPixmap(scaled_pixmap)
        element.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def slider_moved(self, position):
        frame_position = int(position * self.video_processor.frame_count / 100)
        self.video_processor.set_frame_position(frame_position)
        self.update_current_time()

    def item_clicked(self, item):
        frame_position = self.transform_time(item.text()) * self.video_processor.fps
        self.video_processor.set_frame_position(frame_position)
        self.Video_bar.setValue(int(frame_position / self.video_processor.frame_count * 100))

    def toggle_play_pause(self):
        print(self.video_processor.is_playing)
        self.video_processor.is_playing = not self.video_processor.is_playing
        icon = QIcon('Video/icon/play.png' if not self.video_processor.is_playing else 'Video/icon/stop-button.png')
        self.btn_stop_start.setIcon(icon)

    def forward(self):
        self.video_processor.set_frame_position(self.video_processor.current_frame + int(125 / 18 * self.video_processor.fps))
        self.Video_bar.setValue(int(self.video_processor.current_frame / self.video_processor.frame_count * 100))

    def backward(self):
        self.video_processor.set_frame_position(self.video_processor.current_frame - int(125 / 36 * self.video_processor.fps))
        self.Video_bar.setValue(int(self.video_processor.current_frame / self.video_processor.frame_count * 100))

    def update_current_time(self):
        self.Current.setText(self.transform_time(self.video_processor.current_frame / self.video_processor.fps))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.timer.stop()
            self.timer_video.stop()
            self.video_processor.release()
            self.camera_processor.release()
            event.accept()
            sys.exit()
        else:
            event.ignore()

    def play_saved_frames(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        output_path = f'output_video_{timestamp}.mp4'
        frame_delay = 1 / 30

        try:
            self.frame_saver.save_to_video(output_path)
        except ValueError as e:
            print(e)
            return

        for frame in self.frame_saver.frames:
            self.show_img(self.play_frame_view, self.scene3, frame)
            QCoreApplication.processEvents()
            time.sleep(frame_delay)

    def dialog_open(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle('3초 전 영상 재생')
        self.play_frame_view = QGraphicsView(self.dialog)
        self.scene3 = QGraphicsScene()
        self.play_frame_view.setScene(self.scene3)

        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(self.play_frame_view)
        self.dialog.setLayout(dialog_layout)

        self.message_label = QLabel("동영상 저장 중", self.dialog)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("QLabel {font-size: 24px; font-weight: bold; }")
        dialog_layout.addWidget(self.message_label)

        screen = QDesktopWidget().screenGeometry()
        width = screen.width() // 2
        height = screen.height() // 2
        self.dialog.resize(width, height)

        self.dialog.show()
        self.play_saved_frames()
        self.message_label.setText("동영상 저장 완료")

    def muting(self) :
        if self.checkBox2.isChecked() :
            self.checkBox2.setText("On")
        else :
            self.checkBox2.setText("Off")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
