import os
import paramiko
import requests
import json
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.40.128',port='22',username='b793170',password ='02030203')

#파일전송(로컬->서버)
sftp =ssh.open_sftp()
remotepath = '/home/b793170/filename.exe' # sftp에 업로드 될때 파일 경로/파일이름
localpath  = 'c:/Users/Hwang/Desktop/filename.exe' # local pc의 파일 경로/파일이름
sftp.put(localpath, remotepath)

#분석
#stdin, stdout, stderr = ssh.exec_command('cuckoo')#api서버 접속
stdin, stdout, stderr = ssh.exec_command('cuckoo submit --timeout 90 /home/b793170/filename.exe')

#파일존재여부 확인
output = False
result = False
sec = 5

while True :
    stdin, stdout, stderr = ssh.exec_command('[ -f /home/b793170/.cuckoo/storage/analyses/1/reports/report.json ] && echo "$FILE True" || echo "$FILE False"')
    output =''.join(stdout.readlines())
    result = output.replace(" ","")
    json.loads(result.lower())
    time.sleep(sec)
    if json.loads(result.lower()) != False :
        break

#파일전송(서버->로컬)
remotepath2 = '/home/b793170/.cuckoo/storage/analyses/1/reports/report.json'
localpath2 = 'c:/Users/Hwang/Desktop/filename.json'
sftp.get(remotepath2, localpath2)


stdin, stdout, stderr = ssh.exec_command('cuckoo clean')
stdin, stdout, stderr = ssh.exec_command('rm filename.exe')

ssh.close()
sftp.close()