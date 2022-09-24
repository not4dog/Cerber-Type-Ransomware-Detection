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
from PyQt5.QtGui import QIcon
import csv
from datetime import datetime

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
            sleep(0.1)
            self._signal.emit(i)

class OptionWindow(QDialog):    
    def __init__(self, parent):
        super(OptionWindow, self).__init__(parent)       
        option_ui = "Report.ui"        
        uic.loadUi(option_ui, self)     
        self.show()
        self.setWindowTitle('CTRD v1.0 Detection Report')
        self.SaveCSV.clicked.connect(self.CreateReport)

    def CreateReport(self):
        now = datetime.now()
        f = open('Detection_Report\CTRD_Detection_Report.csv','w', newline='')
        wr = csv.writer(f)
        wr.writerow(["<CTRD v1.0 Uploaded File Detection Report>"])
        wr.writerow([])
        wr.writerow(["File Path", filename[0]])
        wr.writerow(["File Size", filesize])
        wr.writerow(["SHA256 Hash", sha256])
        wr.writerow(["Detection Result"])
        wr.writerow(["Scan Date", now.strftime('%Y-%m-%d %H:%M:%S')])
        f.close()
        msgBox = QMessageBox() 
        msgBox.setStyleSheet('QMessageBox {color:black; background:white;}')
        msgBox.information(msgBox,'Notice','CTRD 결과보고서 파일이 생성되었습니다.\n\nDetection_Report 폴더를 확인해 주시기 바랍니다.', msgBox.Ok)

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CTRD v1.0')
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setupUi(self)
        self.Initial.clicked.connect(self.InitialMethod)
        self.Initial.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
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

    def InitialMethod(self):
        msgbox = QMessageBox()
        msgbox.setStyleSheet('QMessageBox {color:black; background:white;}')
        ret = msgbox.question(msgbox,'Question', '초기화 시 기존 분석자료와 결과보고서가 영구적으로 삭제됩니다.\n\n그래도 초기화 하시겠습니까?', msgbox.Yes | msgbox.No)
        if ret == msgbox.Yes:
           QtCore.QCoreApplication.quit()
           QtCore.QProcess.startDetached(sys.executable, sys.argv)
           os.system('rmdir /s /q Detection_Feature_Data & rmdir /s /q Detection_Report')
        else : return

    def pBar(self):
        self.thread = Thread()
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()

    def DataFolderCreate(self):
        dir_path = "Detection_Feature_Data"
        if os.path.isdir(dir_path) != True :
            os.system('mkdir Detection_Feature_Data')
        else : return

    def ReportFolderCreate(self):
        dir_path = "Detection_Report"
        if os.path.isdir(dir_path) != True :
            os.system('mkdir Detection_Report')
        else : return

    def signal_accept(self, msg):
        self.progressBar.setValue(int(msg))
        if self.progressBar.value() == 100:
            self.progressBar.setValue(0)
            self.msg_box()
        return

    def msg_box(self):
        msg = QMessageBox()                      
        ret = msg.information(msg,'Notice', '실행파일 분석이 완료되었습니다.\n\nOk 버튼을 클릭하시면 CTRD 결과보고서 확인이 가능합니다.', msg.Ok | msg.Cancel)
        if ret == msg.Ok:
           OptionWindow(self)
           self.Run.setDisabled(False)
        else : self.Run.setDisabled(False)
        return

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
        return

    def ExtractOpcode(self) :
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
        wr.writerow(["SHA256", "push", "mov", "call", "sub", "jmp", "add", "cmp", "test", "lea", "pop"])
        wr.writerow([sha256, push, mov, call, sub, jmp, add, cmp, test, lea, pop])
        f.close()
        return

    def CountOpcode(self, item):
        file = open("Detection_Feature_Data\File_Opcode_Extract.txt", "r")
        read_data = file.read()
        word_count = read_data.lower().count(item)
        return word_count

    def FileSize(self, size_bytes):
        import math
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def Main(self):
        global filename, filesize
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
               return(print('실행파일이 아닌 파일 선택으로 인한 메인함수 중단'))
               
            file_size = os.path.getsize('{0}' .format(filename[0]))
            filesize = self.FileSize(file_size)
            self.Run.setDisabled(True)
            self.pBar()
            self.ExtractOpcode()

        else :
            msgBox = QMessageBox() 
            msgBox.setStyleSheet('QMessageBox {color:black; background:white;}')
            msgBox.warning(msgBox,'Warning','분석 대상 파일이 선택되지 않았습니다.\n\n파일을 선택해 주시기 바랍니다.')
            return(print('파일 미 선택으로 인한 메인함수 중단'))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.ico'))
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()