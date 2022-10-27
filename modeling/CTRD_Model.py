# 라이브러리 정리 모음

import pandas as pd
import numpy as np  # numpy 1.20 or less
from pathlib import Path
from filepath import get_project_root_directory,check_file_exist

# pycaret -> 오류시 sckit-learn : 0.23.2로 다운그레이드해야 됨
# ( 참고 : 파이썬 3.9버전 이상부터 사용 불가 : https://pycaret.gitbook.io/docs/get-started/installation 참조)

from pycaret.classification import *
from pycaret.utils import check_metric

# 모델이 예측한 확률 값을 직접적으로 반영하여 평가
from sklearn.metrics import log_loss

class AutoML:
    def __init__(self):
        # 가져올 데이터 -> 데이터 프레임 형식으로 변환 
        # 모델 구축 환경 -> 튜플 형식으로 반환
        self.raw_data = pd.DataFrame()
        self.environment = ()
        self.model_path = get_project_root_directory() / Path('modeling')
        self.raw_dataset_path = get_project_root_directory() / Path('GUI/CTRD_Feature_Data/End_Analysis_Data.csv')
        self.uploaded_file_location = get_project_root_directory() / Path('GUI/CTRD_Feature_Data/testing.csv')

    # 데이터 로드
    def load_data(self):
        raw_data = pd.read_csv("EndAnalysis.csv")
        self.raw_data = raw_data

        col_label = self.raw_data['Cerber']

        # 이 부분은 나중에 정적 분석이나 행위 분석만 실시 할 때 vs 정적 + 행위분석 합친거 탐지율 비교 시 사용
        self.opcode_data = self.raw_data.loc[:, "push":"pop"]
        self.api_data = self.raw_data.loc[:, "FindFirstFile":"FindResourceExW"]

        self.opcode_data= pd.concat([self.opcode_data, col_label], axis=1)
        self.api_data = pd.concat([self.api_data, col_label], axis=1)

    # 모델 환경 설정 구축
    def setup_environment(self):
        # 불필요한 특성 제거
        remove_features = ['SHA-256']

        # 모든 특성 -> 리스트 변환 ( Pycaret에 적용되도록 리스트 형태로 변환해야 됨)
        all_features = list(self.raw_data.drop(['Cerber'], axis=1).columns)

        # 모델 환경 설정 구축 ( 파이프라인 설정 )

        # 전처리 설명 ( pycaret preprocessing : https://pycaret.gitbook.io/docs/get-started/preprocessing )
        # 파이프라인 자세한 코드 : https://github.com/shaandinesh/pycaret/blob/master/preprocess.py

        # displot을 보면 skewness 값이 매우 높으며 대부분 데이터 들이 좌측으로 편향되어 있는 것을 확인 할 수 있다.
        # QuantileTransformer 를 통해 정규분포 형태로 변환 ( MinMaxScaler와 비슷하게 0과 1사이로 압축 )

        # feature_interaction : 정적 - 동적 특징들이 상호작용을 가지는 지 확인하기 위해 두 변수를 곱하여 새로운 특성을 만든다. (ex)  * FindFirstFile)
        # feature_selection :  특성 중요도 기반의 유의미하게 사용 할 수 있는 특성 선택을 선택 ( 과적합 ↓ & 정확도 ↑, 훈련시간 ↓ 기대 )
        # 특징-특징과의 상관성 : 0에 가까울 수록 좋고 (interaction), 특징-예측(Cerber) 상관성은 : -1 or 1에 가까울 수 록 좋음  ( 임계값 설정 : 0.01 / 0.9)
        # 선택 알고리즘 Boruta : 모델 예측 시 가장 큰 영향을 미치는 Feature 를 파악  (Wrapper Method : 후진제거식)

        # 훈련 데이터 세트가 대상 클래스의 균등하지 않을 경우 (Oversampling) -> SMOTE로 방지 ( bengin : 443,  Cerber : 389)

        environment = setup(data=self.raw_data,
                            target='Cerber',
                            train_size=0.8,
                            numeric_features=all_features,
                            ignore_features=remove_features,
                            fix_imbalance=True,
                            transformation=True,
                            transformation_method='quantile',
                            feature_interaction=True,
                            interaction_threshold=0.01,
                            feature_selection=True,
                            feature_selection_threshold=0.9,
                            feature_selection_method='boruta',
                            data_split_shuffle=True,
                            data_split_stratify=True,
                            fold_strategy='stratifiedkfold',
                            fold=10,
                            fold_shuffle=True,
                            session_id=42,
                            silent=True)

        # logloss 적용 및 predict_proba로 제출하기 위해 metric 추가
        add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False, target="pred_proba")

        self.environment = environment

    # # 하이퍼 파라미터 튜닝 ( 안해도 될 것 같음 - 정확도가 높음 )
    # def parameter_tuning(self, model_type: str):
    #     model = create_model(model_type)
    #
    #     tuned_model = tune_model(model, optimize='Accuracy', search_library='optuna', n_iter=10)
    #     return tuned_model
        
    # 예측결과 출력
    def evaluation_result(self, model):

        prediction = predict_model(model)

        print('\nEvaluation Results:')
        Accuracy = check_metric(prediction['Cerber'], prediction['Label'], metric='Accuracy')
        print('Accuracy: ', Accuracy)

        Precision = check_metric(prediction['Cerber'], prediction['Label'], metric='Precision')
        print('Prec: ', Precision)

        Recall = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('Recall: ', Recall)

        F1 = check_metric(prediction['Cerber'], prediction['Label'], metric='F1')
        print('F1: ', F1)


    # # 파일 저장
    # def _save_model(self, model, file_name: str):
    #     model_path = self.model_path / Path(file_name)
    #     save_model(model, model_path)

    # 모델 훈련
    def train_model(self):

            # setup 셋팅 가져오기
            self.setup_environment()
            # svm, ridge는 predict_proba 미지원으로 제외

            et = create_model('et')
            lightgbm = create_model('lightgbm')
            rf = create_model('rf')

            # 3개의 모델을 Stacking
            # Meta Model : Logistic Regression
            # 각 분류기의 예측만 사용하기 위해서 restack = False 설정

            stacking = stack_models(
                                estimator_list = [et,lightgbm,rf],
                                meta_model=None,
                                meta_model_fold=10,
                                restack=False,
                                choose_better=True,
                                optimize='logloss'
                                )

            # 모델 평가
            self.evaluation_result(stacking)

            #모델 평가 분석 -> png 파일로 저장 ( 결과보고서에 쓸 것 )
            plot_model(stacking, plot='confusion_matrix', save=True)
            #plot_model(stacking, plot='class_report', save=True)
            #plot_model(stacking, plot='learning',save=True)
            #plot_model(stacking, plot='vc',save=True)
            #plot_model(stacking, plot='rfe', save=True)
            #plot_model(stacking, plot='feature', save=True)

            # 최종 모델 선정
            final_model = finalize_model(stacking)

            # 모델 저장
            save_model(final_model, "CTRD_Label_Model")
            save_config("CTRD_Config_Model.pkl")


if __name__ == "__main__":
        ModelMaker = AutoML()
        ModelMaker.load_data()
        ModelMaker.train_model()