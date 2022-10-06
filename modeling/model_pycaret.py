# 라이브러리 정리 모음
import pandas as pd
import numpy as np

import model_pycaret
from pycaret.classification import *
from pycaret.utils import check_metric

import os
import sys
import joblib
import itertools

from data_transform import DataPreprocessor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class ModelMaker:
    def __init__(self):
        self.raw_data = None
        self.pd_data = None
        # self.data = None # labeled data

        self.X_train= None
        self.X_test= None
        self.y_train= None
        self.y_test= None

        self.final_model= None

    def load_data(self):
        bengin = pd.read_csv("bengin_frequency.csv")
        cerber = pd.read_csv("cerber_frequency.csv")
        df = pd.concat([bengin, cerber])

        self.raw_data = df
        self.pd_data = self.raw_data.drop(['SHA-256'], axis = 1)

    def remove_outlier_based_std(self):
        # 표준점수 기반 이상치 제거
        for i in range(0, len(self.pd_data.iloc[1])):
            self.pd_data.iloc[:, i] = self.pd_data.iloc[:, i].replace(0, np.NaN)  # optional
            self.pd_data = self.pd_data[~(np.abs(self.pd_data.iloc[:, i] - self.pd_data.iloc[:, i].mean()) > (
                        3 * self.pd_data.iloc[:, i].std()))].fillna(0)

    # def prepare_labeled_data(self):
    #     col_label = self.raw_data['family'].reset_index(drop=True)
    #
    #     df_pd_data = pd.DataFrame(self.pd_data)
    #     res = pd.concat([df_pd_data, col_label],axis=1)
    #     self.data = res

    def split_data(self):
            X = self.pd_data.drop('family', axis=1)
            y = self.pd_data['family']

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


            scaler = StandardScaler()
            self.X_train = pd.DataFrame(scaler.fit_transform(x_train))
            self.X_test = pd.DataFrame(scaler.transform(x_test))


            self.y_train = y_train.reset_index(drop=True)
            self.y_test = y_test.reset_index(drop=True)



    def prepare_model(self):
        print('- - - - - - - - - - - - - - - - ')
        print('모델 구축 시작')
        print('- - - - - - - - - - - - - - - - ')

        training_data = pd.concat([self.X_train, self.y_train], axis=1)
        print(training_data)

        exp_clf = setup(data = training_data, target='family', session_id = 42)
        print(exp_clf)

        print('- - - - - - - - - - - - - - - - ')
        print('데이터 셋 학습 시 적합한 모델')
        print('- - - - - - - - - - - - - - - - ')

        best_model = compare_models(sort='Accuracy', n_select=10, fold = 10)
        print(best_model)


        print('- - - - - - - - - - - - - - - - ')
        print('RandomForest ')
        print('- - - - - - - - - - - - - - - - ')

        rf = create_model(estimator='rf', fold=10, probability_threshold=0.5)

        print()
        print(rf)
        print()


        print('- - - - - - - - - - - - - - - - ')
        print('Optuna_Tuning_RandomForest')
        print('- - - - - - - - - - - - - - - - ')

        tuned_rf = tune_model(rf, n_iter=50, optimize='F1', search_library='optuna',search_algorithm='random')

        print()
        print(tuned_rf)
        print()


        print('- - - - - - - - - - - - - - - - ')
        print('RBF SVM')
        print('- - - - - - - - - - - - - - - - ')

        rbfsvm = create_model(estimator='rbfsvm', fold=10, probability_threshold=0.5)

        print()
        print(rbfsvm)
        print()


        print('- - - - - - - - - - - - - - - - ')
        print('Optuna_Tuning_RBFSVM')
        print('- - - - - - - - - - - - - - - - ')

        tuned_rbfsvm = tune_model(rbfsvm, n_iter=50,search_library='optuna', search_algorithm='random')

        print()
        print(tuned_rbfsvm)
        print()

        print('- - - - - - - - - - - - - - - - ')
        print('Naive Bayes')
        print('- - - - - - - - - - - - - - - - ')

        nb = create_model(estimator='nb', fold=10, probability_threshold=0.5)

        print()
        print(nb)
        print()


        print('- - - - - - - - - - - - - - - - ')
        print('Optuna_Tuning_Naive Bayes')
        print('- - - - - - - - - - - - - - - - ')

        tuned_nb = tune_model(nb, n_iter=50, search_library='optuna', search_algorithm='random')

        print()
        print(tuned_nb)
        print()


        print('- - - - - - - - - - - - - - - - ')
        print('Stacking')
        print('- - - - - - - - - - - - - - - - ')

        stacker = stack_models([tuned_rf,tuned_rbfsvm,tuned_nb], meta_model=None, method='auto', fold = 10, choose_better=True, optimize='Accuracy',probability_threshold=0.5)

        print()
        print(stacker)
        print()

        final_model = finalize_model(stacker)
        self.final_model = final_model

        print('- - - - - - - - - - - - - - - - ')
        print('final 모델 생성')
        print('- - - - - - - - - - - - - - - - ')


    def predict_and_evaluate(self):
        print('- - - - - - - - - - - - - - - - ')
        print('final 모델에 대한 성능 평가')
        print('- - - - - - - - - - - - - - - - ')

        prediction_result = predict_model(self.final_model, data=self.X_test)

        eval_Accuracy = check_metric(self.y_test, prediction_result['Label'], metric='Accuracy')
        print('Accuracy: ', eval_Accuracy)
        eval_f1 = check_metric(self.y_test, prediction_result['Label'] ,metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)

    def save_model(self):
        print('- - - - - - - - - - - - - - - - ')
        print('final 모델 저장')
        print('- - - - - - - - - - - - - - - - ')

        save_model(self.final_model, "final model")


if __name__ == "__main__":


    models = ModelMaker()
    models.load_data()
    models.remove_outlier_based_std()
    models.split_data()
    models.prepare_model()
    models.predict_and_evaluate()
    models.save_model()









    

