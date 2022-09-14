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

class ModelType(object):
    def train(self):
        raise (NotImplemented)

    def save(self):
        raise (NotImplemented)


class RandomForestCF(ModelType):
    def __init__(self, datadir):
        self.datadir = datadir
        self.model = None

    # RandomForest
    def train(self):


        def optuna_rf(trial):
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt"]),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }

            classifier_obj = RandomForestClassifier(param)

            score = cross_val_score(classifier_obj, x_train, y_train, n_jobs=-1, cv=10)
            accuracy = score.mean()
            return accuracy

        # train, Prediction
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        # Accuracy
        accuracy = accuracy_score(y_pred, y_test)
        print("RandomForest : ", accuracy)

    def save(self):
        print('[RandomForest] start save')
        # logger.debug(self.model)
        if self.model:
            joblib.dump(self.model, os.path.join(self.datadir, 'RandomForest_model.txt'))
            # load_model = joblib.load('RandomForest_model.txt') #predict method ,
            # load_model.predict(X)
            # logger.debug('[RandomForest] finish save')

class SVM(ModelType):
    def __init__(self, datadir):
        self.datadir = datadir
        self.model = None

    def train(self):

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        def optuna_svm(trial):
            param = {
                "C": trial.suggest_float("C", 1e-10, 1e10, log=True)
            }

            classifier_obj = SVC(C=param, gamma="auto")

            # train, Prediction
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)

            # Accuracy
            accuracy = accuracy_score(y_pred, y_test)
            print("RandomForest : ", accuracy)

    def save(self):
        print('[SVM] start save')
        # logger.debug(self.model)
        if self.model:
            joblib.dump(self.model, os.path.join(self.datadir, 'SVM.txt'))
            # load_model = joblib.load('RandomForest_model.txt') #predict method ,
            # load_model.predict(X)
                # logger.debug('[RandomForest] finish save')


if __name__ == "__main__":
    study_svm = optuna.create_study(direction="maximize")
    study_svm.optimize(optuna_svm, n_trials=100)

    study_rf = optuna.create_study()
    study_rf.optimize(optuna_rf, n_trials=100)

    print("Number of finished trials: {}".format(len(study_rf.trials)))
    print("Number of finished trials: {}".format(len(study_svm.trials)))

    print("Best trial:")
    trial = study_rf.best_trial, study_svm.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
