import pandas as pd
import numpy as np
import csv
import pycaret
import os

from pathlib import Path
from pycaret.classification import *



def get_prediction_results():
        
        # pkl 파일 가져오기
        model = load_model("CTRD_Label_Model")
        
        # 업로드한 파일 가져오기
        upload_data = pd.read_csv('testing.csv')
           
        # 예측 확률 구하기
        Score = model.predict_proba(upload_data)
        
        # Score[0] : 정상일 확률 / Score[1] : Cerber일 확률
        Convert_Benign_Score = round(Score[0] * 100, 2)
        Convert_Cerber_Score = round(Score[1] * 100, 2)
        
        # 임계점 50% 넘으면 Cerber로 간주하여 파일 삭제 (경로 이용)
        if Convert_Cerber_Score > 50.00 :
            os.remove(filepath)
        else :
            pass

        print(Convert_Cerber_Score)
        
        return Convert_Benign_Score, Convert_Cerber_Score


if __name__ == "__main__":
    get_prediction_results()
