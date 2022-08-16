# 참고사이트

# https://github.com/melpin/capstone_design_project_2/blob/master/optuna.py
# https://dacon.io/codeshare/4646
# https://dacon.io/competitions/official/235840/codeshare/3834

# 라이브러리 정리 모음
import numpy as np
import pandas as pd
import optuna
import itertools
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 분류 알고리즘 비교
from numpy import mean
from numpy import std

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# Classification 시각화

def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):
    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1: (len(lines) - 4)]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().replace(' avg', '-avg').split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.figure(figsize=(10, 10))

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.tight_layout()


# Optuna 파라미터 최적화
def objective(trial):
    df = pd.read_csv("modeling/6.6_dataset.csv")

    # SHA-256부분 & 빈도수가 0인 부분 제외
    df = df.drop(['SHA-256'], axis=1)
    mask = df[
        'push' or 'mov' or 'call' or 'sub' or 'jmp' or 'add' or 'cmp' or 'test' or 'lea' or 'pop' or 'FindFirstFile' or 'SearchPathW' or 'SetFilePointer' or 'FindResourceEx' or 'GetFileAttributesW' or 'SetFileAttributesW' or 'SetFilePointerEx' or 'CryptEncrypt' or 'CreateThread' or 'FindResourceExW'].isin(
        [0])
    df = df[~mask]

    X = df[df.columns.difference(['family'])]
    Y = df['family']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=42)

    # 스케일링
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RF'])

    ## 분류기에 따라 다르게 하이퍼 파라미터를 지정, if-elif-else문 이용
    if classifier_name == 'SVC':
        svc_c = trial.suggest_float("C", 1e-10, 1e10, log=True)
        classifier_obj = SVC(C=svc_c, gamma="auto")

    else:
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True),
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        clf = RandomForestClassifier(max_depth = max_depth, n_estimators= n_estimators)

    score = cross_val_score(classifier_obj, X_train, Y_train, n_jobs=-1, cv=10)
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

