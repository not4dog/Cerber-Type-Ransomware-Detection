import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import Qt
import webbrowser
import os

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("GUI.ui")
form_class = uic.loadUiType(form)[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setupUi(self)
        self.Run.clicked.connect(self.ChooseFileAndExtract)
        self.shutdown.clicked.connect(QApplication.instance().quit)
        self.minimize.clicked.connect(self.hideWindow)
        self.github.clicked.connect(lambda: webbrowser.open('https://github.com/not4dog/GP-RansomwareDetection'))

    def ChooseFileAndExtract(self):
        global filename
        filename = QFileDialog.getOpenFileName(self, 'Choose Executable File', 'C:/','Executable File (*.exe)')
        os.system('objdump -d -j .text {0} > File_Opcode_Extract.txt' .format(filename[0]))

    def hideWindow(self):
        self.showMinimized()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
