# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Video.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(698, 631)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.on_air_detection = QtWidgets.QTabWidget(self.centralwidget)
        self.on_air_detection.setGeometry(QtCore.QRect(0, 0, 701, 591))
        self.on_air_detection.setDocumentMode(False)
        self.on_air_detection.setObjectName("on_air_detection")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.graphicsView = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView.setGeometry(QtCore.QRect(10, 60, 511, 401))
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 470, 501, 71))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_pause = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_pause.setObjectName("btn_pause")
        self.gridLayout.addWidget(self.btn_pause, 1, 0, 1, 1)
        self.btn_stop_start = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_stop_start.setObjectName("btn_stop_start")
        self.gridLayout.addWidget(self.btn_stop_start, 1, 3, 1, 1)
        self.Video_bar = QtWidgets.QSlider(self.gridLayoutWidget)
        self.Video_bar.setOrientation(QtCore.Qt.Horizontal)
        self.Video_bar.setObjectName("Video_bar")
        self.gridLayout.addWidget(self.Video_bar, 0, 2, 1, 2)
        self.Total_length = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Total_length.setObjectName("Total_length")
        self.gridLayout.addWidget(self.Total_length, 0, 4, 1, 1)
        self.Current = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Current.setObjectName("Current")
        self.gridLayout.addWidget(self.Current, 0, 0, 1, 1)
        self.btn_forward = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_forward.setObjectName("btn_forward")
        self.gridLayout.addWidget(self.btn_forward, 1, 4, 1, 1)
        self.btn_prev = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_prev.setObjectName("btn_prev")
        self.gridLayout.addWidget(self.btn_prev, 1, 2, 1, 1)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(530, 10, 160, 531))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Video_info = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.Video_info.setObjectName("Video_info")
        self.verticalLayout.addWidget(self.Video_info)
        self.Video_info_text = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.Video_info_text.setObjectName("Video_info_text")
        self.verticalLayout.addWidget(self.Video_info_text)
        self.Log = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.Log.setObjectName("Log")
        self.verticalLayout.addWidget(self.Log)
        self.Log_text = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.Log_text.setObjectName("Log_text")
        self.verticalLayout.addWidget(self.Log_text)
        self.Video_upload = QtWidgets.QPushButton(self.tab)
        self.Video_upload.setGeometry(QtCore.QRect(10, 20, 91, 23))
        self.Video_upload.setObjectName("Video_upload")
        self.progressBar = QtWidgets.QProgressBar(self.tab)
        self.progressBar.setGeometry(QtCore.QRect(110, 20, 401, 20))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.on_air_detection.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.groupBox = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox.setGeometry(QtCore.QRect(60, 40, 521, 101))
        self.groupBox.setObjectName("groupBox")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 20, 521, 80))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.checkBox1 = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.checkBox1.setObjectName("checkBox1")
        self.gridLayout_2.addWidget(self.checkBox1, 0, 1, 1, 1)
        self.checkBox2 = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.checkBox2.setObjectName("checkBox2")
        self.gridLayout_2.addWidget(self.checkBox2, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_2.setGeometry(QtCore.QRect(60, 160, 521, 101))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_2)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(0, 20, 521, 80))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        self.checkBox3 = QtWidgets.QCheckBox(self.gridLayoutWidget_3)
        self.checkBox3.setObjectName("checkBox3")
        self.gridLayout_3.addWidget(self.checkBox3, 0, 1, 1, 1)
        self.checkBox4 = QtWidgets.QCheckBox(self.gridLayoutWidget_3)
        self.checkBox4.setObjectName("checkBox4")
        self.gridLayout_3.addWidget(self.checkBox4, 1, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 1, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_3.setGeometry(QtCore.QRect(60, 290, 521, 71))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.groupBox_3)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(0, 20, 521, 46))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget_4)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout_4.addWidget(self.comboBox, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 1, 0, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.gridLayoutWidget_4)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.gridLayout_4.addWidget(self.comboBox_2, 1, 1, 1, 1)
        self.on_air_detection.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.btn_start_detection = QtWidgets.QPushButton(self.tab_3)
        self.btn_start_detection.setGeometry(QtCore.QRect(10, 460, 681, 41))
        font = QtGui.QFont()
        font.setFamily("한컴 고딕")
        font.setPointSize(14)
        self.btn_start_detection.setFont(font)
        self.btn_start_detection.setObjectName("btn_start_detection")
        self.Log_2 = QtWidgets.QLabel(self.tab_3)
        self.Log_2.setGeometry(QtCore.QRect(530, 23, 158, 21))
        self.Log_2.setObjectName("Log_2")
        self.Log_text_2 = QtWidgets.QListWidget(self.tab_3)
        self.Log_text_2.setGeometry(QtCore.QRect(530, 50, 158, 401))
        self.Log_text_2.setObjectName("Log_text_2")
        self.on_air_camera = QtWidgets.QGraphicsView(self.tab_3)
        self.on_air_camera.setGeometry(QtCore.QRect(10, 50, 511, 401))
        self.on_air_camera.setObjectName("on_air_camera")
        self.Log_4 = QtWidgets.QLabel(self.tab_3)
        self.Log_4.setGeometry(QtCore.QRect(10, 20, 158, 31))
        self.Log_4.setObjectName("Log_4")
        self.btn_play_3sec = QtWidgets.QPushButton(self.tab_3)
        self.btn_play_3sec.setGeometry(QtCore.QRect(530, 510, 161, 31))
        self.btn_play_3sec.setObjectName("btn_play_3sec")
        self.on_air_detection.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 698, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.on_air_detection.setCurrentIndex(2)
        self.comboBox.setCurrentIndex(0)
        self.comboBox_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_pause.setText(_translate("MainWindow", "▶️"))
        self.btn_stop_start.setText(_translate("MainWindow", "⏯️"))
        self.Total_length.setText(_translate("MainWindow", "00:00:00"))
        self.Current.setText(_translate("MainWindow", "00:00:00"))
        self.btn_forward.setText(_translate("MainWindow", "⏩"))
        self.btn_prev.setText(_translate("MainWindow", "⏪"))
        self.Video_info.setText(_translate("MainWindow", "영상 정보"))
        self.Log.setText(_translate("MainWindow", "로그 "))
        self.Video_upload.setText(_translate("MainWindow", "동영상 업로드"))
        self.on_air_detection.setTabText(self.on_air_detection.indexOf(self.tab), _translate("MainWindow", "기본 화면"))
        self.groupBox.setTitle(_translate("MainWindow", "Log"))
        self.label.setText(_translate("MainWindow", "로그 기록"))
        self.checkBox1.setText(_translate("MainWindow", "Off(Save in TXT)"))
        self.checkBox2.setText(_translate("MainWindow", "Off"))
        self.label_4.setText(_translate("MainWindow", "뮤트(Mute)"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Graphic"))
        self.label_5.setText(_translate("MainWindow", "랙 경계선 표시( Set Border Lines)"))
        self.checkBox3.setText(_translate("MainWindow", "Off"))
        self.checkBox4.setText(_translate("MainWindow", "Off"))
        self.label_6.setText(_translate("MainWindow", "랙 구분(상,하  Rack Classification Mode)"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Video"))
        self.label_7.setText(_translate("MainWindow", "  위험상황 기준 전후 상황 기록 "))
        self.comboBox.setCurrentText(_translate("MainWindow", "±3초"))
        self.comboBox.setPlaceholderText(_translate("MainWindow", "5"))
        self.comboBox.setItemText(0, _translate("MainWindow", "±3초"))
        self.comboBox.setItemText(1, _translate("MainWindow", "±5초"))
        self.comboBox.setItemText(2, _translate("MainWindow", "±7초"))
        self.label_8.setText(_translate("MainWindow", "  위험상황 기준 전후 상황 기록 "))
        self.comboBox_2.setCurrentText(_translate("MainWindow", "5초"))
        self.comboBox_2.setPlaceholderText(_translate("MainWindow", "5"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "5초"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "10초"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "15초"))
        self.on_air_detection.setTabText(self.on_air_detection.indexOf(self.tab_2), _translate("MainWindow", "설정 화면"))
        self.btn_start_detection.setText(_translate("MainWindow", "탐지 시작"))
        self.Log_2.setText(_translate("MainWindow", "로그 "))
        self.Log_4.setText(_translate("MainWindow", "실시간 카메라"))
        self.btn_play_3sec.setText(_translate("MainWindow", "3초 전 영상 재생"))
        self.on_air_detection.setTabText(self.on_air_detection.indexOf(self.tab_3), _translate("MainWindow", "실시간 탐지"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
