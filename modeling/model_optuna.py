# 라이브러리 정리 모음
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import joblib
import optuna
import itertools

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 분류 알고리즘 비교
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


trial = 50;

class Objective_RF():
    def __init__(self):
        self.raw_data = None
        self.pd_data = None

        self.data = None # labeled data

        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
    def load_data(self, cleaned_data):
        bengin = pd.read_csv("bengin_frequency.csv")
        cerber = pd.read_csv("cerber_frequency.csv")
        df = pd.concat([bengin, cerber])
        self.raw_data = df
        self.pd_data = cleaned_data # standardized data 활용

    def split_data(self):
            X = self.data.drop('family', axis=1)
            y = self.data['family']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

    def __call__(self,trial):

            param = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt"]),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }

            classifier_obj = RandomForestClassifier(param)

            score = cross_val_score(classifier_obj, self.X_train, self.y_train, n_jobs=-1, cv=10)
            accuracy = score.mean()
            return accuracy

    def random_forest_tuning(self):
        objective = Objective_RF(self.X_train, self.y_train)
        study = optuna.create_study()

        study.optimize(objective, timeout=180)

        print("Params: ")
        for key, value in trial.params.items():
            print("{}: {}".format(key, value))

        print("Best trial: ".format(study.best_trial))

        print("Best params: ".format(study.best_params))

        print("Value: {}".format(trial.value))

        return study

    def random_forest_learning(self,study):

        rf_best = study.best_params
        clf_rf = RandomForestClassifier(**rf_best)
        clf_rf.fit(self.X_train, self.y_train)

        print('[RandomForest] start save....')

        # 모델 검증
        scorer = make_scorer(accuracy_score)
        print(scorer(clf_rf, self.X_test, self.y_test))



