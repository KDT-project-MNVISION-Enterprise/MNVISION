import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextBrowser, QVBoxLayout, QWidget, QPushButton

class TextBrowserExample(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QTextBrowser Example")
        self.setGeometry(100, 100, 800, 600)

        # Create a widget for the window contents
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create a layout for the widgets
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        # Create the QTextBrowser
        self.text_browser = QTextBrowser(self)
        self.layout.addWidget(self.text_browser)

        # Create buttons
        self.set_text_button = QPushButton("Set Text", self)
        self.set_text_button.clicked.connect(self.set_text)
        self.layout.addWidget(self.set_text_button)

        self.get_text_button = QPushButton("Get Text", self)
        self.get_text_button.clicked.connect(self.get_text)
        self.layout.addWidget(self.get_text_button)

    def set_text(self):
        self.text_browser.setText("Hello, this is a sample text.")

    def get_text(self):
        text = self.text_browser.toPlainText()
        print("Current text:", text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = TextBrowserExample()
    example.show()
    sys.exit(app.exec_())
