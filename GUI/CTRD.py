import sys
from xml.dom.minidom import parseString
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import webbrowser
import os
import hashlib
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
import csv
from datetime import datetime
import paramiko
import json
from time import *
from PyQt5.QtTest import *
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import gspread

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("GUI.ui")
form_class = uic.loadUiType(form)[0]
form2 = resource_path("Report.ui")
spreadKey = resource_path("CTRD_Upload_Spread.json")

class Thread(QThread):
    _signal = pyqtSignal(int)
    def __init__(self):
        super(Thread, self).__init__()
    
    def run(self):
        for i in range(101):
            sleep(2)
            self._signal.emit(i)

class OptionWindow(QDialog):    
    def __init__(self, parent):
        super(OptionWindow, self).__init__(parent)  
        self.ui = uic.loadUi(form2, self)         
        self.show()
        self.setWindowTitle('CTRD v1.0 Detection Report')
        self.FilePath.setText(filename[0])
        self.FileSize.setText(filesize)
        self.Hash.setText(sha256)
        self.Result.setText("test%")
        self.SaveCSV.clicked.connect(self.CreateReport)

    def CreateReport(self):
        now = datetime.now()
        f = open(f'CTRD_Report\{sha256}_CTRD_Detection_Report.csv','w', newline='')
        wr = csv.writer(f)
        wr.writerow(["<CTRD v1.0 Uploaded File Detection Report>"])
        wr.writerow([])
        wr.writerow(["File Path",'', filename[0]])
        wr.writerow(["File Size",'', filesize])
        wr.writerow(["SHA256 Hash",'', sha256])
        wr.writerow(["Detection Result",''])
        wr.writerow(["Scan Date",'', now.strftime('%Y-%m-%d %H:%M:%S')])
        f.close()
        msgBox = QMessageBox() 
        msgBox.setStyleSheet('QMessageBox {color:black; background:white;}')
        msgBox.information(msgBox,'Notice','CTRD 결과보고서 파일이 생성되었습니다.\n\nCTRD_Report 폴더를 확인해 주시기 바랍니다.', msgBox.Ok)

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
        self.github.clicked.connect(lambda: webbrowser.open('https://github.com/not4dog/Cerber-Type-Ransomware-CTRD'))
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
           os.system('rmdir /s /q CTRD_Feature_Data & rmdir /s /q CTRD_Report')
        else : return

    def pBar(self):
        self.thread = Thread()
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()

    def DataFolderCreate(self):
        dir_path = "CTRD_Feature_Data"
        if os.path.isdir(dir_path) != True :
            os.system('mkdir CTRD_Feature_Data')
        else : return

    def ReportFolderCreate(self):
        dir_path = "CTRD_Report"
        if os.path.isdir(dir_path) != True :
            os.system('mkdir CTRD_Report')
        else : return

    def signal_accept(self, msg):
        self.progressBar.setValue(int(msg))
        if self.progressBar.value() == 100:
            self.progressBar.setValue(0)
            self.msg_box()
        return

    def msg_box(self):
        msg = QMessageBox()                      
        ret = msg.information(msg,'Notice', '실행파일 분석이 완료되었습니다.\n\nOK 버튼을 클릭하시면 CTRD 결과보고서 확인이 가능합니다.', msg.Ok | msg.Cancel)
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

    def CountAPI(self, item):
        file = open(f"CTRD_Feature_Data\{sha256}_API_Extract.json", "r")
        read_data = file.read()
        word_count = read_data.lower().count(item)
        return word_count

    def ExtractOpcode(self) :
        self.CheckSHA256()
        os.system('objdump -d -j .text {0} > CTRD_Feature_Data\{1}_Opcode_Extract.txt' .format(filename[0], sha256))
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

        f = open(f'CTRD_Feature_Data\{sha256}_Opcode_Frequency.csv','w', newline='')
        wr = csv.writer(f)
        wr.writerow(["SHA256", "push", "mov", "call", "sub", "jmp", "add", "cmp", "test", "lea", "pop"])
        wr.writerow([sha256, push, mov, call, sub, jmp, add, cmp, test, lea, pop])
        f.close()
        return

    def CountOpcode(self, item):
        file = open(f"CTRD_Feature_Data\{sha256}_Opcode_Extract.txt", "r")
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
    
    def reset(self):
        loop = QEventLoop()
        QTimer.singleShot(5000, loop.quit) 
        loop.exec_()

    def sshConnect(self):
        global ssh
        QApplication.processEvents()
        ssh = paramiko.SSHClient()
        QApplication.processEvents()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        QApplication.processEvents()
        ssh.connect('211.214.61.14',port='2200',username='b793170',password ='20100709')
        QApplication.processEvents()

    def FileTransper(self):
        global sftp
        sftp =ssh.open_sftp()
        QApplication.processEvents()
        remotepath = '/home/b793170/Desktop/Scan.exe' 
        localpath  = filepath 
        QApplication.processEvents()
        sftp.put(localpath, remotepath)
        QApplication.processEvents()

    def Analysis(self):
        stdin, stdout, stderr = ssh.exec_command('curl -H "Authorization: Bearer pxJLRqiTfxz0PNNhGLdoew" -F file=@/home/b793170/Desktop/Scan.exe http://localhost:8090/tasks/create/file')
        QApplication.processEvents()

    def Exists(self):
        output = False
        result = False

        while True :
            stdin, stdout, stderr = ssh.exec_command('[ -f /home/b793170/.cuckoo/storage/analyses/1/reports/report.json ] && echo "$FILE True" || echo "$FILE False"')
            output =''.join(stdout.readlines())
            result = output.replace(" ","")
            json.loads(result.lower())
            QApplication.processEvents()
            self.reset()
            QApplication.processEvents()
            if json.loads(result.lower()) != False :
                break
            QApplication.processEvents()

    def FileTransperAndExtract(self):
        remotepath2 = '/home/b793170/.cuckoo/storage/analyses/1/reports/report.json'
        localpath2 = 'CTRD_Feature_Data\{0}_API_Extract.json' .format(sha256)
        sftp.get(remotepath2, localpath2)
        QApplication.processEvents()
        stdin, stdout, stderr = ssh.exec_command("rm -f /home/b793170/Desktop/Scan.exe")
        QApplication.processEvents()
        stdin, stdout, stderr = ssh.exec_command('curl -H "Authorization: Bearer pxJLRqiTfxz0PNNhGLdoew" http://localhost:8090/tasks/delete/1')
        QApplication.processEvents()
        ssh.close()
        QApplication.processEvents()
        sftp.close()
        QApplication.processEvents()

        api1 = self.CountAPI("findfirstfile")
        api2 = self.CountAPI("searchpathw")
        api3 = self.CountAPI("setfilepointer")
        api4 = self.CountAPI("findresourceex")
        api5 = self.CountAPI("getfileattributesw")
        api6 = self.CountAPI("setfileattributesw")
        api7 = self.CountAPI("setfilepointerex")
        api8 = self.CountAPI("cryptencrypt")
        api9 = self.CountAPI("createthread")
        api10 = self.CountAPI("findresourceexw")

        f = open(f'CTRD_Feature_Data\{sha256}_API_Frequency.csv','w', newline='' .format(sha256))
        wr = csv.writer(f)
        wr.writerow(["FindFirstFile", "SearchPathW", "SetFilePointer", "FindResourceEx", "GetFileAttributesW", "SetFileAttributesW", "SetFilePointerEx", "CryptEncrypt", "CreateThread", "FindResourceExW"])
        wr.writerow([api1, api2, api3, api4, api5, api6, api7, api8, api9, api10])
        f.close()
        return

    def sleep(self):
        QTest.qWait(5000)

    def FeatureMerge(self):
        Opcode = 'CTRD_Feature_Data/{0}_Opcode_Frequency.csv' .format(sha256)
        api = 'CTRD_Feature_Data/{0}_API_Frequency.csv' .format(sha256)

        dataFrame = pd.concat(map(pd.read_csv, [Opcode, api]), axis=1)
        dataFrame.to_csv(r'CTRD_Feature_Data/All_Feature_CTRD_Data.csv', index = False)

    def UploadSpread(self):
        scope = ["https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/drive"]

        creds = ServiceAccountCredentials.from_json_keyfile_name(spreadKey, scope)
        QApplication.processEvents()

        spreadsheet_name = "CTRD_Feature_Data"
        client = gspread.authorize(creds)
        QApplication.processEvents()
        spreadsheet = client.open(spreadsheet_name)
        QApplication.processEvents()

        for sheet in spreadsheet.worksheets():
            QApplication.processEvents()
            sheet

        new_df = pd.read_csv('CTRD_Feature_Data/All_Feature_CTRD_Data.csv')
        QApplication.processEvents()
        val_list = new_df.values.tolist()
        load_list =val_list[0]

        sheet.append_row(load_list)
        QApplication.processEvents()

    def Main(self):
        global filename, filesize
        global filepath
        filename = QFileDialog.getOpenFileName(self, 'Choose Executable File', 'C:/','Executable File (*.exe)') 
        filepath = filename[0]

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

            self.pBar()
            self.Run.setDisabled(True)
            self.ExtractOpcode() 
            self.sshConnect()
            self.FileTransper()
            self.Analysis()
            self.Exists()
            self.FileTransperAndExtract()
            self.FeatureMerge()
            self.UploadSpread()

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