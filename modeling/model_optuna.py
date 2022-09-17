# 라이브러리 정리 모음
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import joblib,pickle
import optuna
import itertools

from data_transform import DataPreprocessor

from sklearn.pipeline import make_pipeline


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Normalizer

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,StackingClassifier


class FinalModel:
    def __init__(self):
            bengin = pd.read_csv("bengin_frequency.csv")
            cerber = pd.read_csv("cerber_frequency.csv")
            df = pd.concat([bengin, cerber])
            self.raw_data = df

            self.pd_data = self.raw_data.drop(['SHA-256'], axis=1)

            self.X = self.pd_data.drop('family', axis=1)
            self.y = self.pd_data['family']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)

            self.m_models = []

            self.m_strModels = ['Logisitic Regression', 'Random Forest', 'Decision Tree', 'Svm', 'Naive Bayes']


    def objective_LR(self, trial):
        param_lr = {
            "C" : trial.suggest_loguniform("C", 1e-5, 1e5),
            "solver" : trial.suggest_categorical("solver", ("lbfgs", "saga")),
            "max_iter" : trial.suggest_int("max_iter", 4000, 4000)
        }

        pipe = make_pipeline(StandardScaler(), LogisticRegression(**param_lr))

        pipe.fit(self.X_train, self.Y_train)

        return pipe.score(self.X_test, self.Y_test)



    def objective_rf(self,trial):
            param_rf = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt"]),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
            }

            classifier_rf = RandomForestClassifier(**param_rf)

            score = cross_val_score(classifier_rf, self.X_train, self.y_train, n_jobs=-1, cv=10)
            accuracy = score.mean()
            return accuracy

    def objective_svc(self,trial):
        param_svc = {
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "gamma": trial.suggest_loguniform('gamma', 1e-5, 1e5)
        }

        classifier_svc = SVC(**param_svc)

        score = cross_val_score(classifier_svc, self.X_train, self.y_train, n_jobs=-1, cv=10)
        accuracy = score.mean()
        return accuracy


    def ApplyLogisticRegression(self,showSteps):
        if showSteps is True:
            print('Starting logistic regression optimization.')



    def ApplyRandomForest(self,showSteps):

        if showSteps is True:
            print('Starting random forest optimization.')

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_rf, n_trials= 50 ,timeout=180)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial: ".format(study.best_trial))

        print("Best params: ".format(study.best_params))

        rf_best = study.best_params

        clf_rf = RandomForestClassifier(**rf_best)

        clf_rf.fit(self.X_train, self.y_train)

        # 모델 저장
        print('[RandomForest] start save....')
        joblib.dump(clf_rf, 'RF_model.pkl')

        # 모델 검증
        scorer = make_scorer(accuracy_score)
        print(scorer(clf_rf, self.X_test, self.y_test))

        y_pred = clf_rf.predict(self.X_test)

        # 모델 예측
        print(f"accuracy : {accuracy_score(self.y_test, y_pred) * 100 :.3f}")
        print(f"precision : {precision_score(self.y_test, y_pred) * 100 : .3f}")
        print(f"recall : {recall_score(self.y_test, y_pred) * 100 : .3f}")
        print(f"f1 : {f1_score(self.y_test, y_pred) * 100 :.3f}")

        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

        if showSteps is True:
            print('Decision tree optimized.')

        return clf_rf

    def ApplySVM(self, showSteps):
        if showSteps is True:
            print('Starting svm optimization.')

        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective_svc, n_trials=50, timeout=180)

        trial = study.best_trial

        C = trial.params["C"]
        kernel = trial.params["kernel"]

        svc = SVC(C=C, kernel=kernel, probability=True)
        svc.fit(self.X_train, self.y_train)

        if showSteps is True:
            print('Svm optimized.')

        return svc

    def CoreModelTraining(self, nameOfCaller, showSteps=False):
        if showSteps is True:
            print(f'---{nameOfCaller} model training beginning.---')


        self.m_models.append(('Logistic Regression', self.ApplyLogisticRegression(showSteps)))
        self.m_models.append(('Random Forest', self.ApplyRandomForest(showSteps)))
        self.m_models.append(('Decision Tree', self.ApplyDecisionTree(showSteps)))
        self.m_models.append(('SVM', self.ApplySVM(showSteps)))
        self.m_models.append(('Naive Bayes', self.ApplyNaiveBayes(showSteps)))

        if showSteps is True:
            print(f'---{nameOfCaller} model training done.---')

    def TrainModelsStacking(self, showSteps=False):
        self.CoreModelTraining('Stacking', showSteps)

        # Initialize the stacked classifier and train it.
        sc = StackingClassifier(estimators=self.m_models, final_estimator=LogisticRegression(max_iter=4000))
        sc.fit(self.X_train, self.y_train)

        xTestPredictions = sc.predict(self.X_test)
        print(f'Accuracy score of stacking classifier:\n{accuracy_score(self.y_test, xTestPredictions)}')



if __name__ == '__main__':











