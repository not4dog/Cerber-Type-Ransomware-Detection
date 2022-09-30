import paramiko
import json
import time
from GUIOpen import *

def sshConnect():
    global ssh
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('211.214.61.14',port='22',username='ubuntu',password ='ubuntu')
    return

def UploadedFileTransper():
    global sftp 
    sftp = ssh.open_sftp()
    remotepath = '/home/ubuntu/Uploaded_exe/Scan.exe' 
    localpath  = filename[0]
    sftp.put(localpath, remotepath)
    return

def CuckooAnalysis():
    stdin, stdout, stderr = ssh.exec_command('cuckoo')
    stdin, stdout, stderr = ssh.exec_command('cuckoo submit --timeout 90 /home/ubuntu/Uploaded_exe/Scan.exe')
    return

def jsonTransper():
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

    stdin, stdout, stderr = remotepath2 = '/home/ubuntu/.cuckoo/storage/analyses/1/report.json'
    stdin, stdout, stderr = localpath2 = 'Detection_Feature_Data/{0}_API_Extract.json' .format(sha256)
    stdin, stdout, stderr = sftp.get(remotepath2, localpath2)
    return

def AnalysisFileRemove():
    stdin, stdout, stderr = ssh.exec_command('rm -f /home/ubuntu/Uploaded_exe/Scan.exe')
    stdin, stdout, stderr = ssh.exec_command('cuckoo clean')
    return

def Exit():
    ssh.close()
    sftp.close()
    return


