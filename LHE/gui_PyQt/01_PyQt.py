import sys #명령행 인수를 GUI 에 전달할 수 있도록
from PyQt5.QtWidgets import QApplication, QWidget
 
app = QApplication(sys.argv)    # 추가 명령행 인수를 전달할 수 있도록 sys.argv로 전달해 인스턴스 생성하여 변수에 저장
gui = QWidget()                 # Qwidget 인스턴스 생성 -> GUI가 됨.
gui.setWindowTitle('First PyQt GUI')
gui.show()                      # GUI를 볼 수 있도록 show메서드 호출
sys.exit(app.exec_())           # exec_() 메서드 : 애플리케이션 실행.
# 예외 사항을 알기 위해  호출을 sys.exit() 안에 매핑.