import os
import paramiko
import requests
import json

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.40.128',port='22',username='b793170',password ='02030203')

#파일전송(로컬->서버)
sftp =ssh.open_sftp()
remotepath = '/home/b793170/Desktop/5db631589b179544445b62d72e5021293408799c694c848452752a8a01517d2e.exe' # sftp에 업로드 될때 파일 경로/파일이름
localpath  = 'c:/Users/Hwang/Desktop/5db631589b179544445b62d72e5021293408799c694c848452752a8a01517d2e.exe' # local pc의 파일 경로/파일이름
sftp.put(localpath, remotepath)

#분석
stdin, stdout, stderr = ssh.exec_command('cuckoo api') #api서버 접속
stdin, stdout, stderr = ssh.exec_command("curl -H 'Authorization: Bearer pxJLRqiTfxz0PNNhGLdoew' -F 'file=@/home/b793170/Desktop/5db631589b179544445b62d72e5021293408799c694c848452752a8a01517d2e.exe' http://localhost:8090/tasks/create/file")
stdin, stdout, stderr = ssh.exec_command('curl -H "Authorization: Bearer pxJLRqiTfxz0PNNhGLdoew" http://localhost:8090/tasks/report/2/json >5db631589b179544445b62d72e5021293408799c694c848452752a8a01517d2e.json')


#결과
print(''.join(stdout.readlines()))


#파일전송(서버->로컬)
remotepath2 = '/home/b793170/5db631589b179544445b62d72e5021293408799c694c848452752a8a01517d2e.json'
localpath2 = 'c:/Users/Hwang/Desktop/5db631589b179544445b62d72e5021293408799c694c848452752a8a01517d2e.json'
sftp.get(remotepath2, localpath2)

ssh.close()
sftp.close()