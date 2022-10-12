# 라이브러리 정리 모음

import pandas as pd
import numpy as np  # numpy 1.20 or less
from pathlib import Path
from filepath import get_project_root_directory,check_file_exist

# pycaret -> 오류시 sckit-learn : 0.23.2로 다운그레이드해야 됨 ( 참고 : 파이썬 3.9버전 이상부터 0.23.2로 변경 안 됨, )
from pycaret.classification import *
from pycaret.utils import check_metric
from sklearn.metrics import log_loss # 모델이 예측한 확률 값을 직접적으로 반영하여 평가

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
        # 먼저 Min-Max 스케일링으로 0~1로 범위를 좁혀 정규화 시킨다
        # 그 다음 비선형변환 방식인 yeo-johnson 사용하여 정규 분포 형식으로 데이터를 고르게 분포

        # 특성 중요도 기반의 유의미하게 사용 할 수 있는 특성 선택을 선택 ( 과적합 ↓ & 정확도 ↑, 룬련시간 ↓ 기대 )
        # 특징-특징과의 상관성 : 0에 가까울 수록 좋고, 특징-예측(Cerber) 상관성은 : -1 or 1에 가까울 수 록 좋음 (기본값 사용 : 0.8 )
        # 선택 알고리즘 permutation importance : 모델 예측에 가장 큰 영향을 미치는 Feature 를 파악

        # 훈련 데이터 세트가 대상 클래스의 균등하지 않을 경우 (Oversampling) -> SMOTE로 방지 ( bengin : 443,  Cerber : 389)

        environment = setup(data=self.raw_data,
                            target='Cerber',
                            train_size=0.8,
                            numeric_features=all_features,
                            ignore_features=remove_features,
                            normalize=True,
                            normalize_method='robust',
                            feature_interaction=True,
                            feature_selection=True,
                            feature_selection_threshold=0.9,
                            feature_selection_method='classic',
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

        F1 = check_metric(prediction['Cerber'], prediction['Label'], metric='Recall')
        print('F1: ', F1)

        return Accuracy,Precision,Recall,F1

    # # 파일 저장
    # def _save_model(self, model, file_name: str):
    #     model_path = self.model_path / Path(file_name)
    #     save_model(model, model_path)

    # 모델 훈련
    def train_model(self):

            # 환경 셋팅
            self.setup_environment()

            # 모델 추천  ( LogLoss가 낮은 모델을 선정해야됨 - gbc,rf,et )
            # svm, ridge는 predict_proba 미지원으로 제외 & lightgbm은 스태킹 모델에서 사용되므로 제외
            best3 = compare_models(fold=10, sort='logloss', n_select=3, exclude=['svm', 'ridge'])

            # 4개의 모델을 Stacking
            stacking = stack_models(
                                estimator_list = best3,
                                meta_model=None,
                                meta_model_fold=10,  # 내부 메타 모델 교차 검증 제어
                                method="predict_proba",
                                choose_better=True,
                                optimize="logloss"
                                )

            # 모델 평가
            self.evaluation_result(stacking)

            #모델 평가 분석 -> png 파일로 저장 ( 결과보고서에 쓸 것 )
            # plot_model(stacking, plot='auc', save=True)
            # plot_model(stacking, plot='pr', save=True)
            # plot_model(stacking, plot='learning', save=True)
            # plot_model(stacking, plot='threshold', save=True)
            # plot_model(stacking, plot='confusion_matrix', save=True)
            # plot_model(stacking, plot='class_report', save=True)

            # 최종 모델 선정
            final_model = finalize_model(stacking)

            # 모델 저장
            save_model(final_model, "CTRD_model")
            save_config("CTRD_config.pkl")


if __name__ == "__main__":
        ModelMaker = AutoML()
        ModelMaker.load_data()
        ModelMaker.train_model()