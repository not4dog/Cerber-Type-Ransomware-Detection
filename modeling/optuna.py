# 참고사이트

# https://github.com/melpin/capstone_design_project_2/blob/master/optuna.py
# https://dacon.io/codeshare/4646
# https://dacon.io/competitions/official/235840/codeshare/3834

# 라이브러리 정리 모음
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,joblib,optuna
import itertools
import pickle

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
    """
    Train the RandomForest model from the vectorized features
    """
    def __init__(self, datadir, rows, dim):
        self.datadir = datadir
        self.rows = rows
        self.dim = dim
        self.model = None

    # RandomForest
    def train(self):
        """
        Train
        """
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        self.model = RandomForestClassifier(n_estimators=100,
                                            min_samples_leaf=25,
                                            max_features=0.5,
                                            n_jobs=-1,
                                            oob_score=False)

        # 파라미터 튜닝 : https://injo.tistory.com/30 참조

        # train, Prediction
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        # Accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_pred, y_test)
        print("RandomForest : ", accuracy)

    def save(self):
        """
        Save a model using a pickle package
        """
        print('[RandomForest] start save')
        # logger.debug(self.model)
        if self.model:
            joblib.dump(self.model, os.path.join(self.datadir, 'RandomForest_model.txt'))
            # load_model = joblib.load('RandomForest_model.txt') #predict method ,
            # load_model.predict(X)
            # logger.debug('[RandomForest] finish save')


# Optuna 파라미터 최적화
def objective(trial):

    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RF'])

    ## 분류기에 따라 다르게 하이퍼 파라미터를 지정, if-else문 이용
    if classifier_name == 'SVC':
        svc_c = trial.suggest_float("C", 1e-10, 1e10, log=True)
        classifier_obj = SVC(C=svc_c, gamma="auto")

    else:
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True),
        n_estimators = trial.suggest_int('n_estimators', 100, 1000),
        max_features = trial.suggest_categorical("max_features", ["auto", "sqrt"]),

        classifier_obj = RandomForestClassifier(max_depth = max_depth, n_estimators= n_estimators, max_features = max_features)

    score = cross_val_score(classifier_obj, X_train, Y_train, n_jobs=-1, cv=10)
    accuracy = score.mean()
    return accuracy



if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



    

