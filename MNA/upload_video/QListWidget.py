import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QVBoxLayout, QWidget, QMessageBox

class ListWidgetExample(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QListWidget Example")
        self.setGeometry(100, 100, 400, 300)

        # Create a widget for the window contents
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create a layout for the widgets
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        # Create the QListWidget
        self.list_widget = QListWidget(self)
        self.layout.addWidget(self.list_widget)

        # Add items to the QListWidget
        for i in range(10):
            self.list_widget.addItem(f"Item {i+1}")

        # Connect the itemClicked signal to a slot
        self.list_widget.itemClicked.connect(self.item_clicked)

    def item_clicked(self, item):
        QMessageBox.information(self, "Item Clicked", f"You clicked: {item.text()}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = ListWidgetExample()
    example.show()
    sys.exit(app.exec_())
