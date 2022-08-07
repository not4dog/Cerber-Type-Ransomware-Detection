# 참고사이트

# https://github.com/melpin/capstone_design_project_2/blob/master/optuna.py
# https://dacon.io/codeshare/4646
# https://dacon.io/competitions/official/235840/codeshare/3834

import pandas as pd
import optuna

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Optuna 파라미터 최적화
def objective(trial):

    df = pd.read_csv("modeling/6.6_dataset.csv")
    X = df[df.columns.difference(['family'])]
    Y = df['family']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])

    ## 분류기에 따라 다르게 하이퍼 파라미터를 지정, if-elif-else문 이용
    if classifier_name == 'SVC':
        C = trial.suggest_loguniform('C', 1e-4, 1e4),
        # gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)
        classifier_obj = SVC(C = C, gamma= 'auto')

    elif classifier_name == 'RandomForest':
        max_depth = trial.suggest_int('max_depth', 1, 10),
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000),
        n_estimators : trial.suggest_int('n_estimators', 100, 500)
        classifier_obj = RandomForestClassifier(max_depth = max_depth, n_estimators= n_estimators)

    accuracy = cross_val_score(classifier_obj, x, y, cv=4).mean()
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))