import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import webbrowser
from time import *
import os
from PyQt5 import QtCore
from PyQt5 import QtGui

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("GUI.ui")
form_class = uic.loadUiType(form)[0]

class Thread(QThread):
    _signal = pyqtSignal(int)
    def __init__(self):
        super(Thread, self).__init__()
    
    def run(self):
        for i in range(100):
            sleep(2)
            self._signal.emit(i)
        
class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setupUi(self)
        self.Run.clicked.connect(self.ChooseFileAndExtract)
        self.Run.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shutdown.clicked.connect(QApplication.instance().quit)
        self.minimize.clicked.connect(self.hideWindow)
        self.github.clicked.connect(lambda: webbrowser.open('https://github.com/not4dog/Cerber-Type-Ransomware-Detection'))
        self.github.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def pBar(self):
        self.thread = Thread()
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()

    def signal_accept(self, msg):
        self.progressBar.setValue(int(msg))
        if self.progressBar.value() == 99:
            self.progressBar.setValue(0)
            self.msg_box()

    def hideWindow(self):
        self.showMinimized()

    def msg_box(self):
        msg = QMessageBox()                      
        msg.information(msg,'Notice','Executable File Analysis is Complete.\nPlease Check the Report Folder.')                               

    def ChooseFileAndExtract(self):
        global filename
        filename = QFileDialog.getOpenFileName(self, 'Choose Executable File', 'C:/','Executable File (*.exe)')
        if filename[0] !='' :
            self.pBar()
            os.system('objdump -d -j .text {0} > File_Opcode_Extract.txt' .format(filename[0]))
            
        else :
            msgBox = QMessageBox() 
            msgBox.setStyleSheet('QMessageBox {color:black; background:white;}')
            msgBox.warning(msgBox,'Warning','Please Select a Executable File.')
            self.ChooseFileAndExtract()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
