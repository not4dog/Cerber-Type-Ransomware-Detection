import pandas as pd
import numpy as np
import csv

from pathlib import Path
from csv import reader
from filepath import get_project_root_directory
from pycaret.classification import *
from pycaret.utils import check_metric


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

def load_automl_model(model_file_name: str):
    model_path = model_location / Path(model_file_name)
    model = load_model(model_path)

    return model

def automl_analyze_file_n_predict():

    # 업로드한 파일의 csv 가져오기
    upload_data= pd.read_csv(uploaded_file_location)

    # 모델 pkl 가져오기
    model = load_automl_model('final_model')

    # 딕셔너리 형식으로 결과기 만들기 ( key-value )
    prediction_results = dict()

    raw_prediction = predict_model(model, data=upload_data)

    prediction_results['CTRD_Model'] = dict()
    
    # 파일 최소 2개 이상이여야 함 ( 정상 케르베르 )
    prediction_results['CTRD_Model']['benign'] = raw_prediction['Label'][0]
    prediction_results['CTRD_Model']['cerber'] = raw_prediction['Label'][1]
    prediction_results['CTRD_Model']['benign_score'] = '%.2f%%' % (raw_prediction['Score'][0]*100)
    prediction_results['CTRD_Model']['cerber_score'] = '%.2f%%' % (raw_prediction['Score'][1]*100)

    #  결과 출력
    print(prediction_results)
    # 예측 결과 리턴
    return prediction_results


if __name__ == "__main__":
    automl_analyze_file_n_predict()