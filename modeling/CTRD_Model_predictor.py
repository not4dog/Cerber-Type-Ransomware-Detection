import pandas as pd
import numpy as np
import csv
import pycaret


from pathlib import Path
from filepath import get_project_root_directory
from CTRD_Model import AutoML

from pycaret.classification import *
from pycaret.utils import *
from sklearn.metrics import log_loss # 모델이 예측한 확률 값을 직접적으로 반영하여 평가


# 파일 경로 설정
uploaded_file_location  = get_project_root_directory() / Path('GUI/CTRD_Feature_Data/testing.csv')
model_location= get_project_root_directory() / Path('modeling')

# 업로드 한 파일 경로를 통해서 업로드한 실행 파일을 가져오고 삭제 기능으로 만들 수 있지 않을까?
def load_file(file_path):
    with open(file_path, 'rb') as file_load:
        file_content = file_load.read()
        return file_content

def delete_file(file_path):
    Path(file_path).unlink()


def get_prediction_results():

    # 업로드한 파일의 csv 가져오기
    upload_data= pd.read_csv(uploaded_file_location)

    # 모델 pkl 가져오기
    model = load_model('CTRD_model')

    # # 예측 결과를 딕셔너리 형태로 표현
    # prediction_results = dict()

    # 업로드 한 파일의 csv에 대해서 구축한 탐지 모델에 넣어 탐지 확률을 구한다. > raw_score = predict_proba 기능!
    raw_prediction = predict_model(model, data=upload_data, raw_score=True)
    print(raw_prediction.head)

    # # 예측 딕셔너리에 결과값을 추가
    # prediction_results['model'] = dict()

    # Score 0 : 모델에서 업로드 한 파일이 정상으로 판정한 확률 / Score 1 : 모델에서 업로드 한 파일이 Cerber로 판정한 확률
    # prediction_results['model']['benign'] = '%.2f%%' % (raw_prediction['Score_0'][0]*100)
    # prediction_results['model']['Cerber'] = '%.2f%%' % (raw_prediction['Score_1'][0]*100)
    #
    # print(prediction_results)

    # 클래스에 대한 확률 값을 출력
    bengin_score = raw_prediction['Score_0']*100
    cerber_score = raw_prediction['Score_1']*100

    print("\n")
    print(bengin_score)
    print(cerber_score)

    # 업로드한 파일 예측 결과 csv 형태로 변환 ( 미완 )
    # raw_prediction.to_csv("파일경로명")


if __name__ == "__main__":
    get_prediction_results()