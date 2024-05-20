import sys
from PyQt5.QtWidgets import QApplication, QWidget

class GUI(QWidget):                 # Qwidget 상속하는 새 클래스 생성.
    def __init__(self): 
        super().__init__()          # super를 호출하고 차례로 GUI를 만든다
        self.initUI()               # refer to Window as 'self'
          
    def initUI(self):   
        self.setWindowTitle('PyQt5 GUI')    # call method
 
 
if __name__ == '__main__':     
    app = QApplication(sys.argv)        # 추가 명령행 인수를 전달할 수 있도록 sys.argv로 전달해 인스턴스 생성하여 변수에 저장    
    gui = GUI()                         # 인스턴스 생성
    gui.show()                          # GUI를 볼 수 있도록 show메서드 호출
    sys.exit(app.exec_())               # exec_() 메서드 : 애플리케이션 실행.
