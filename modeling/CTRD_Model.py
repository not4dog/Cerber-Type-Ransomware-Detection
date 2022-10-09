# 라이브러리 정리 모음

import pandas as pd
import numpy as np  # numpy 1.20 or less
from pathlib import Path
from file import get_project_root_directory,check_file_exist

# pycaret -> 오류시 sckit-learn : 0.23.2로 다운그레이드해야 됨 ( 참고 : 파이썬 3.9버전 이상부터 0.23.2로 변경 안 됨,  3.7,3.8 버전으로 바꿔서 해야 될 듯)
from pycaret.classification import *
from pycaret.distributions import UniformDistribution, DiscreteUniformDistribution, CategoricalDistribution, IntUniformDistribution
from pycaret.utils import check_metric


class AutoML:
    def __init__(self):
        # 경로명 지정해줘야 됨
        self.model_path = get_project_root_directory() / Path('modeling')
        self.raw_dataset_path = get_project_root_directory() / Path('All_Feature_CTRD_Analysis.csv')
        self.raw_data = pd.DataFrame()
        self.environment = ()
    
    # 데이터 로드
    def load_data(self):
        raw_data = pd.read_csv("ALL_Feature_CTRD_Analysis.csv")
        self.raw_data = raw_data
        raw_data.to_csv(self.raw_dataset_path)

        col_label = self.raw_data['Cerber']

        # 이 부분은 나중에 정적 분석이나 행위 분석만 실시 할 때 vs 정적 + 행위분석 합친거 탐지율 비교 시 사용
        self.opcode_data = self.raw_data.loc[:, "push":"pop"]
        self.api_data = self.raw_data.loc[:, "FindFirstFile":"FindResourceExW"]

        self.opcode_data= pd.concat([self.opcode_data, col_label], axis=1)
        self.api_data = pd.concat([self.api_data, col_label], axis=1)
    
    # 모델 환경 설정 구축
    def setup(self):

        # 모든 특성 -> 리스트 변환 ( Pycaret에 적용되도록 리스트 형태로 변환해야 됨)
        all_features = list(self.raw_data.drop(['SHA-256','Cerber'], axis=1).columns)

        # 불필요한 특성 제거
        remove_features = ['SHA-256']


        environment = setup(data=self.raw_data,
                            target="Cerber",
                            train_size=0.8,
                            numeric_features=all_features,
                            ignore_features=remove_features,
                            normalize=True,
                            normalize_method="robust", # 이상치 영향을 최소화를 위해 RobustScaler 스케일링
                            feature_selection=True, # 특성 중요도 기반의 유의미하게 사용 할 수 있는 특성 선택 ( 기본 알고리즘 : SelectFromModel )
                            fix_imbalance=True, # 불균형 방지를 위해 SMOTE 적용
                            fold_strategy="stratifiedkfold",
                            fold=10,
                            fold_shuffle=True,
                            session_id=42,
                            silent=True # 자동화
                            )
        self.environment = environment

    # 하이퍼 파라미터 튜닝
    def parameter_tuning(model_type: str):
        model = create_model(model_type)

        tuned_model = tune_model(model, optimize='Accuracy', search_library='optuna', n_iter=10)
        return tuned_model
        
        # 예측결과 출력
    def evaluation_result(model):
        prediction = predict_model(model)

        Accuracy = check_metric(prediction['Cerber'], prediction['Label'], metric='Accuracy')
        print('Accuracy: ', Accuracy)

        Precision = check_metric(prediction['Cerber'], prediction['Label'], metric='Precision')
        print('Prec: ', Precision)

        Recall = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('Recall: ', Recall)

        F1 = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('F1: ', F1)

        return prediction,Accuracy,Precision,Recall,F1
    
    # 모델 저장
    def save_model(self, model, file_name: str):
        model_path = self.model_path / Path(file_name)
        save_model(model, model_path)
    
    # 모델 훈련
    def train_model(self, model_name: str):
        # 데이터 셋 없을 시 경로에서 가져오기
            if not check_file_exist(self.raw_dataset_path):
                self.load_data()
            else:
                self.raw_data = pd.read_csv(self.raw_dataset_path)

            # RandomForest, R
            rf_model = self.parameter_tuning('rf')
            lightgbm_model = self.parameter_tuning('lightgbm')
            gbc_model = self.parameter_tuning('gbc')

            stacking = stack_models(
                                estimator_list=[rf_model, lightgbm_model, gbc_model],
                                meta_model=None,  # 기본적으로 로지스틱 회귀 사용
                                meta_model_fold=10,  # 내부 메타 모델 교차 검증 제어
                                fold=10,
                                method="auto",
                                choose_better=True,
                                optimize="Accuracy",
                                )
            # 최종 모델 평가
            self.evaluation_result(stacking)

            # 모델 평가 분석 -> png 파일로 저장 ( 결과보고서에 쓸 것 )
            plot_model(stacking, plot='auc', save=True)
            plot_model(stacking, plot='threshold', save=True)
            plot_model(stacking, plot='confusion_matrix', save=True)
            plot_model(stacking, plot='learning', save=True)

            # 최종 모델 결정
            final = finalize_model(stacking)

            # 모델 저장
            self.save_model(final, model_name)


if __name__ == "__main__":
        ModelMaker = AutoML()
        ModelMaker.load_data()
        ModelMaker.setup()
        ModelMaker.train_model("rf_model")