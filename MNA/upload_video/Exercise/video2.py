from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import time

class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super(VideoPlayer, self).__init__(parent)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()
        videoItem = QGraphicsVideoItem()
        # videoItem.setWindowFlag(Qt.FramelessWindowHint)
        videoItem.setSize(QSizeF(self.width(), self.height()))
        scene = QGraphicsScene(self)
        scene.addItem(videoItem)
        graphicsView = QGraphicsView(scene)
        layout = QVBoxLayout()
        layout.addWidget(graphicsView)
        self.setLayout(layout)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(videoItem)

        # stateChanged 시그널 연결
        self.mediaPlayer.stateChanged.connect(self.handleStateChanged)

    def keyPressEvent(self, e):
        """단축키"""
        print('state: ' + str(self.mediaPlayer.state()))
        print('mediaStatus: ' + str(self.mediaPlayer.mediaStatus()))
        print('error: ' + str(self.mediaPlayer.error()))
        if e.key() == Qt.Key_L: # L키는 로드
            print('loading')
            self.load()
        if e.key() == Qt.Key_P: # P키는 플레잉
            print('playing')
            self.mediaPlayer.play()
        if e.key() == Qt.Key_Q: # Q키 누르면 종료
            self.close()
        else:
            return

    def load(self):
        """wmv 파일 로드하기"""
        local = QUrl.fromLocalFile('./person.avi')
        media = QMediaContent(local)
        self.mediaPlayer.setMedia(media)

    def handleStateChanged(self, state):
        """비디오가 종료되면 화면도 꺼지게 하는 함수"""
        if state == QMediaPlayer.StoppedState:
            self.close()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())