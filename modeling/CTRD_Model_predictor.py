import pandas as pd
import numpy as np

# 모델링 부분 불러옴
from CTRD_Model import AutoML

#
from pathlib import Path
from filepath import get_project_root_directory,check_file_exist
from pycaret.classification import *

# 파일 위치 선정
uploaded_file_location = get_project_root_directory() / Path('GUI/CTRD')
model_location = get_project_root_directory() / Path('modeling')

#
def load_automl_model(model_file_name: str):
    model_path = model_location / Path(model_file_name)
    model = load_model(model_path)
    return model

