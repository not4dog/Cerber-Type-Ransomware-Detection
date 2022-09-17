# 라이브러리 정리 모음
import pandas as pd
import numpy as np

import model_pycaret
from pycaret.classification import *
from pycaret.utils import check_metric
from pycaret.datasets import get_data

import os
import sys
import joblib
import itertools
import optuna

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class ModelMaker:
    def __init__(self):
        self.raw_data = None
        self.pd_data = None
        self.data = None # labeled data

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

    def remove_unnecessary_features(self):
        self.pd_data = self.raw_data.drop(['SHA-256'], axis=1)

    def split_data(self):
            X = self.pd_data.drop('family', axis=1)
            y = self.pd_data['family']

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            self.X_train = x_train
            self.X_test = x_test
            self.y_train = y_train
            self.y_test = y_test


    def prepare_model(self):
        print('- - - - - - - - - - - - - - - - ')
        print('모델 구축 시작')
        print('- - - - - - - - - - - - - - - - ')


        training_data = pd.concat([self.X_train, self.y_train], axis=1)
        print(training_data)

        exp_clf = setup(data = training_data, target='family', session_id = 42)

        rf = create_model(estimator='rf', fold=10, probability_threshold=0.5)

        print('rf: ')
        print()
        print(rf)
        print()

        tuned_rf = tune_model(rf, n_iter=10, optimize='F1', search_library='optuna', search_algorithm='random', choose_better=True)

        print('tuned_rf: ')
        print()
        print(tuned_rf)
        print()

        rbfsvm = create_model(estimator='rbfsvm', fold=10, probability_threshold=0.5)

        print('rbfsvm: ')
        print()
        print(rbfsvm)
        print()

        tuned_rbfsvm = tune_model(rbfsvm, n_iter=50, optimize='F1', search_library='optuna', search_algorithm='random', choose_better=True)

        print('tuned_rbfsvm: ')
        print()
        print(tuned_rbfsvm)
        print()

        stacker = stack_models([tuned_rf, tuned_rbfsvm],meta_model=None, method='auto', fold = 10, choose_better=True, optimize='Accuracy',probability_threshold=0.5)

        print('stacker: ')
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
        eval_f1 = check_metric( self.y_test, prediction_result['Label'] ,metric='F1')
        print('F1: ', eval_f1)
        eval_prec = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_prec)
        eval_re = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_re)

        print()
        print()
    def save_model(self):
        print('- - - - - - - - - - - - - - - - ')
        print('final 모델 저장')
        print('- - - - - - - - - - - - - - - - ')

        save_model(self.final_model, "final model")


if __name__ == "__main__":
    models = ModelMaker()
    models.load_data()
    models.remove_unnecessary_features()
    models.split_data()
    models.prepare_model()
    models.predict_and_evaluate()
    models.savemodel()







    

