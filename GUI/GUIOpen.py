import sys
from xml.dom.minidom import parseString
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import webbrowser
from time import *
import os
import hashlib
from PyQt5 import QtCore
from PyQt5 import QtGui
import csv

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
        for i in range(101):
            sleep(2)
            self._signal.emit(i)
        
class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setupUi(self)
        self.DataFolderCreate()
        self.ReportFolderCreate()
        self.Run.clicked.connect(self.Main)
        self.Run.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shutdown.clicked.connect(QApplication.instance().quit)
        self.minimize.clicked.connect(self.hideWindow)
        self.github.clicked.connect(lambda: webbrowser.open('https://github.com/not4dog/Cerber-Type-Ransomware-Detection'))
        self.hongiklogo.clicked.connect(lambda: webbrowser.open('https://sejong.hongik.ac.kr/index.do'))
        self.github.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.hongiklogo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.minimize.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shutdown.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def pBar(self):
        self.thread = Thread()
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()

    def DataFolderCreate(self):
        dir_path = "Detection_Feature_Data"
        if os.path.isdir(dir_path) != True :
            os.system('mkdir Detection_Feature_Data')
        else : pass

    def ReportFolderCreate(self):
        dir_path = "Detection_Report"
        if os.path.isdir(dir_path) != True :
            os.system('mkdir Detection_Report')
        else : pass

    def signal_accept(self, msg):
        self.progressBar.setValue(int(msg))
        if self.progressBar.value() == 100:
            self.progressBar.setValue(0)
            self.msg_box()

    def msg_box(self):
        msg = QMessageBox()                      
        msg.information(msg,'Notice','실행파일 분석이 완료되었습니다.\n\nDetection_Report 폴더에서 탐지 결과를 확인해 주시기 바랍니다.')

    def hideWindow(self):
        self.showMinimized()
    
    def CheckSHA256(self):
        global sha256
        hash_sha256 = hashlib.sha256()
        with open(filename[0], "rb") as f:
            chunk = f.read(4096)
            while chunk:
                hash_sha256.update(chunk)
                chunk = f.read(4096)
        sha256 = hash_sha256.hexdigest()

    def CountOpcode(self, item):
        file = open("Detection_Feature_Data\File_Opcode_Extract.txt", "r")
        read_data = file.read()
        word_count = read_data.lower().count(item)
        return word_count

    def Main(self):
        global filename
        filename = QFileDialog.getOpenFileName(self, 'Choose Executable File', 'C:/','Executable File (*.exe)') 

        if filename[0] !='' :
            with open(filename[0], 'rb') as f:
                signature1 = f.read(4)
                signature2 = signature1
    
            if signature1 == b'MZ\x90\x00' or signature2 == b'MZP\x00' :
               pass

            else :
               msgBox = QMessageBox() 
               msgBox.setStyleSheet('QMessageBox {color:black; background:white;}')
               msgBox.warning(msgBox,'Warning','선택한 파일은 실행파일이 아닙니다.\n\n올바른 실행파일을 선택해 주시기 바랍니다.')
               return

            self.pBar()
            os.system('objdump -d -j .text {0} > Detection_Feature_Data\File_Opcode_Extract.txt' .format(filename[0]))
            push = self.CountOpcode("push")
            mov = self.CountOpcode("mov")
            call = self.CountOpcode("call")
            sub = self.CountOpcode("sub")
            jmp = self.CountOpcode("jmp")
            add = self.CountOpcode("add")
            cmp = self.CountOpcode("cmp")
            test = self.CountOpcode("test")
            lea = self.CountOpcode("lea")
            pop = self.CountOpcode("pop")
            self.CheckSHA256()

            f = open('Detection_Feature_Data\Opcode_Item_Frequency.csv','w', newline='')
            wr = csv.writer(f)
            wr.writerow([sha256, push, mov, call, sub, jmp, add, cmp, test, lea, pop,])
            f.close()

        else :
            msgBox = QMessageBox() 
            msgBox.setStyleSheet('QMessageBox {color:black; background:white;}')
            msgBox.warning(msgBox,'Warning','분석할 파일이 선택되지 않았습니다.\n\n파일을 선택해 주시기 바랍니다.')
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()