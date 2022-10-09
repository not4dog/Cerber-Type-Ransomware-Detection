# 라이브러리 정리 모음
import pandas as pd
import numpy as np
from pathlib import Path
from file import get_project_root_directory,check_file_exist

# pycaret -> 오류시 sckit-learn : 0.23.2로 다운그레이드해야 됨
from pycaret.classification import *
from pycaret.distributions import UniformDistribution, DiscreteUniformDistribution, CategoricalDistribution, IntUniformDistribution
from pycaret.utils import check_metric



class AutoML:
    def __init__(self):
        # 경로명 지정해줘야 됨
        self.model_path = get_project_root_directory() / Path('modeling')
        self.raw_dataset_path = get_project_root_directory() / Path('GUI/CTRD_Feature_Data/All_Feature_CTRD_Data.csv')
        self.raw_data, self.opcode_data, self.api_data = pd.DataFrame()
        self.environment = ()
    
    # 데이터 로드
    def load_data(self):
        raw_data = self.raw_data
        raw_data.to_csv(self.raw_dataset_path)

        col_label = self.raw_data['Cerber']

        # 이 부분은 나중에 정적 분석이나 행위 분석만 실시 할 때 vs 정적 + 행위분석 합친거 탐지율 비교 시 사용
        self.opcode_data = self.raw_data.loc[:, "push":"pop"]
        self.api_data = self.raw_data.loc[:, "FindFirstFile":"FindResourceExW"]

        self.opcode_data= pd.concat([self.opcode_data, col_label], axis=1)
        self.api_data = pd.concat([self.api_data, col_label], axis=1)
    
    # 환경 설정 구축
    def setup(self):
        remove_features = ['SHA-256']
        all_features = list(self.raw_data.drop(['SHA-256','Cerber'], axis=1).columns)

        environment = setup(data=self.raw_data,
                            target="Cerber",
                            train_size=0.8,
                            numeric_features=all_features,
                            ignore_features=remove_features,
                            normalize=True,
                            normalize_method="robust",
                            fix_imbalance=True,
                            feature_selection=True,
                            fold_strategy="kfold",
                            fold=10,
                            fold_shuffle=True,
                            session_id=42
                            )
        self.environment = environment
        
        # 모델 추천
        best_model = compare_models(sort='Accuracy', fold=10, n_select=10, Budget_time=1)
        print(best_model)

        # 하이퍼 파라미터 튜닝
    def parameter_tuning(model_type: str):
        model = create_model(model_type)

        params ={
                "n_estimators": IntUniformDistribution(100, 250),
                "max_depth": IntUniformDistribution(1, 10),
                "min_samples_split": IntUniformDistribution(2, 10),
                "min_samples_leaf": IntUniformDistribution(1, 10),
                "C": UniformDistribution(1e-5, 1e5),
                "gamma": UniformDistribution(1e-5, 1e5),
            }
        
        tuned_model = tune_model(model, optimize='Accuracy',custom_grid= params, search_library='optuna', n_iter=30)
        return tuned_model
        
        # 예측결과 출력
    def evaluation_result(model):
        prediction = predict_model(model)

        Accuracy = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('Accuracy: ', Accuracy)

        Precision = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('Prec: ', Precision)

        recall = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('Recall: ', recall)

        f1 = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('F1: ', f1)

        return prediction,Accuracy,Precision,recall,f1
    
    # 모델 저장
    def save_model(self, model, file_name: str):
        model_path = self.model_path / Path(file_name)
        save_model(model, model_path)
    
    # 모델 훈련
    def train_model(self, model_name: str):
        # Get the dataset, either by extracting software files or reading CSV file
            if not check_file_exist(self.raw_dataset_path):
                self.load_data()
            else:
                self.raw_data = pd.read_csv(self.raw_dataset_path)

            # RandomForest, R
            rf_model = self.parameter_tuning('rf')
            svm_model = self.parameter_tuning('rbfsvm')
            nb_model = self.parameter_tuning('nb')


            stacking = stack_models(estimator_list=[rf_model, svm_model, nb_model],
                                meta_model=None,  # 기본적으로 로지스틱 회귀 사용
                                meta_model_fold=10,  # 내부 메타 모델 교차 검증 제어
                                fold=10,
                                method="auto",
                                choose_better=True,
                                optimize="Accuracy",
                                probability_threshold=0.5 # 예측 확률을 클래스 레이블로 변환
                                )
            # 최종 모델 평가
            self.evaluation_result(stacking)

            # 모델 성능 분석 -> png 파일로 저장 ( 결과보고서에 쓸 것 )
            plot_model(stacking, plot='pipeline', save=True)
            plot_model(stacking, plot='auc', save=True)
            plot_model(stacking, plot='threshold', save=True)
            plot_model(stacking, plot='confusion_matrix', save=True)
            plot_model(stacking, plot='learning', save=True)


            # 최종 모델
            combined_final = finalize_model(stacking)

            # 모델 저장
            self.save_model(combined_final, model_name)