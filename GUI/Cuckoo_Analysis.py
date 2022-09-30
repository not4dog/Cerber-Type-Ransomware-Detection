import paramiko
import json
from time import *
from CTRD import *

def sshConnect():
    global ssh
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('211.214.61.14',port='2200',username='b793170',password ='20100709')

#파일전송(로컬->서버)
def FileTransper():
    global sftp
    sftp =ssh.open_sftp()
    remotepath = '/home/b793170/Desktop/Scan.exe' # sftp에 업로드 될때 파일 경로/파일이름
    localpath  = filepath # local pc의 파일 경로/파일이름
    sftp.put(localpath, remotepath)

#분석
#stdin, stdout, stderr = ssh.exec_command('cuckoo')#api서버 접속
def Analysis():
    stdin, stdout, stderr = ssh.exec_command("cuckoo")
    stdin, stdout, stderr = ssh.exec_command("cuckoo submit --timeout 90 /home/b793170/Desktop/Scan.exe")

#파일존재여부 확인
def Exists():
    output = False
    result = False
    sec = 5

    while True :
        stdin, stdout, stderr = ssh.exec_command('[ -f /home/b793170/.cuckoo/storage/analyses/1/reports/report.json ] && echo "$FILE True" || echo "$FILE False"')
        output =''.join(stdout.readlines())
        result = output.replace(" ","")
        json.loads(result.lower())
        sleep(sec)
        if json.loads(result.lower()) != False :
            break

#파일전송(서버->로컬)
def FileTransperAndRemove():
    remotepath2 = '/home/b793170/.cuckoo/storage/analyses/1/reports/report.json'
    localpath2 = 'Detection_Feature_Data\{0}_API_Extract.json' .format(sha256)
    sftp.get(remotepath2, localpath2)

    stdin, stdout, stderr = ssh.exec_command("cuckoo clean")
    stdin, stdout, stderr = ssh.exec_command("rm -f /home/b793170/Desktop/Scan.exe")

    ssh.close()
    sftp.close()
