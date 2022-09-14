import json
import urllib
import sys
import os
import time
import re
import hashlib

_md5 = '[a-z0-9]{32}'

# 랜섬웨어, 정상파일 폴더를 만들고 연습용 실행파일 폴더 설정
path_dir = './bengin'
# path_dir = './modeling/cerber'
file_list = os.listdir(path_dir)

md5_pattern = re.compile(_md5)


# 파일의 MD5로 바이러스토탈 DB 검색 해 기존 분석 이력있으면 분석 결과 가져오기
class vtAPI():
    def __init__(self):
        self.api = '310939e7baa50b87c52a5ede74184c0ff2019aa7add61e66b436661b5ee6c91b'
        self.base = 'https://www.virustotal.com/vtapi/v2/'

    def getReport(self, md5):
        param = {'resource': md5, 'apikey': self.api, 'allinfo': '1'}
        url = self.base + "file/report"
        data = urllib.urlencode(param)
        result = urllib.urlopen(url, data)

        jdata = json.loads(result.read())

        if jdata['response_code'] == 0:
            print(md5 + " -- Not Found in VT")
            return "no"
        else:
            print("=== Results for MD5: ", jdata['md5'], "\tDetected by: ", jdata['positives'])
            return jdata['positives']

    # 파일 업로드 후 분석 요청
    def reqScan(self, filepath):
        print("- Requesting a new scan")
        param = {'file': filepath, 'apikey': self.api}
        url = self.base + "file/scan"
        data = urllib.urlencode(param)
        result = urllib.urlopen(url, data)

        jdata = json.loads(result.read())

        return jdata

    # 분석 대상 파일의 MD5 값을 구하는 함수
    def getMd5(self, filepath, blocksize=8192):
        md5 = hashlib.md5()
        try:
            f = open(filepath, "rb")
        except IOError as e:
            print("file open error", e)
            return
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            md5.update(buf)
        return md5.hexdigest()


def main():
    vt = vtAPI()
    i = 0

    for file in file_list:

        before = path_dir + "/" + file
        name_check = re.search(md5_pattern, file)

        if name_check == None:
            file = vt.getMd5(before)

        try:
            i += 1
            rns = vt.getReport(file)
            if (rns == "no"):
                file_path = os.getcwd() + "/" + file
                rns = vt.reqScan(file_path)
                file = rns['md5']

                while True:
                    time.sleep(30)
                    rns = vt.getReport(file)
                    if (rns != "no"):
                        break

            after = path_dir + "/" + str(rns) + "#" + file

            print("Processed " + str(i) + " files - " + after)
            os.rename(before, after)

            time.sleep(15)
        except:
            pass


if __name__ == '__main__':
    main()