# 라이브러리 정리 모음
import pandas as pd

from pycaret.classification import *
from pycaret.utils import check_metric

from sklearn.model_selection import train_test_split

from optuna import Trial, visualization
from data_transform import DataPreprocessor

class ModelMaker:
    def __init__(self):
        self.opcode_data = None
        self.api_data = None

        self.raw_data = None
        self.pd_data = None

        self.data = None # labeled data

        self.X_train= None
        self.X_test= None
        self.y_train= None
        self.y_test= None

        self.final_model= None

    def load_data(self, cleaned_data):

        self.raw_data = pd.read_csv("All_Feature_CTRD_Data.csv")
        self.pd_data = cleaned_data # standardized data 활용

        # 이 부분은 나중에 정적 분석이나 행위 분석만 실시 할 때 vs 정적 + 행위분석 합친거 탐지율 비교 시 사용
        self.opcode_data = self.raw_data.loc[:,"push":"pop"]
        self.api_data = self.raw_data.loc[:,"FindFirstFile":"FindResourceExW"]


        print(self.raw_data)
        #print(self.opcode_data)
        #print(self.api_data)


    def prepare_labeled_data(self):
        col_label = self.raw_data['Cerber']

        self.opcode_data= pd.concat([self.opcode_data, col_label], axis=1)
        self.api_data = pd.concat([self.api_data, col_label], axis=1)

        df_pd_data = pd.DataFrame(self.pd_data)
        res = pd.concat([df_pd_data, col_label], axis=1)
        self.data = res

    def split_data(self):
            # Train & Test 분리
            y = self.data['Cerber']
            X = self.data.drop(['Cerber'],axis=1)



            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            self.X_train = x_train
            self.X_test = y_test
            self.y_train = y_train
            self.y_test = y_test

    def prepare_model(self):

        # 모델 구축
        training_data = pd.concat([self.X_train, self.y_train], axis=1)
        print(training_data)

        # 모델
        s = setup(training_data,
                  target='Cerber',
                  train_size=0.8,
                  fold_strategy='stratifiedkfold',
                  fold=10,
                  fix_imbalance=True,
                  feature_selection=True)

        # 데이터 셋에 따른 적합한 모델 선정

        best_model = compare_models(sort='Accuracy', n_select=10, fold = 10)
        print(best_model)

        print('- - - - - - - - - - - - - - - - ')
        print('RandomForest')
        print('- - - - - - - - - - - - - - - - ')

        # RandomForest
        rf = create_model(estimator='rf', fold=10, probability_threshold=0.5)

        print()
        print(rf)
        print()

        print('- - - - - - - - - - - - - - - - ')
        print('Optuna_Tuning_RandomForest')
        print('- - - - - - - - - - - - - - - - ')

        # 하이퍼 파라미터 튜닝

        # rf_params = {
        #     "n_estimators": Trial.suggest_int("n_estimators", 10, 100),
        #     "max_depth": Trial.suggest_int("max_depth", 1, 10),
        #     "min_samples_split": Trial.suggest_int("min_samples_split", 2, 10),
        #     "min_samples_leaf": Trial.suggest_int("min_samples_leaf", 1, 5)
        # }

        tuned_rf = tune_model(rf, n_iter=20, optimize='Accuracy', search_library='optuna', search_algorithm = "tpe")

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

        # svm_params = {
        #         "C": Trial.suggest_loguniform("C", 1e-5, 1e5),
        #         "gamma" : Trial.suggest_loguniform('gamma',1e-5,1e5),
        # }

        tuned_rbfsvm = tune_model(rbfsvm, n_iter=20, optimize='Accuracy', search_library='optuna', search_algorithm = "tpe")

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
        print('Stacking')
        print('- - - - - - - - - - - - - - - - ')

        stacking = stack_models(estimator_list=[tuned_rf,tuned_rbfsvm,nb],
                                meta_model = None, # 기본적으로 로지스틱 회귀 사용
                                meta_model_fold=10, # 내부 메타 모델 교차 검증 제어
                                fold=10,
                                method="auto",
                                choose_better=True,
                                optimize="Accuracy"
                                )

        print()
        print(stacking)
        print()

        print('- - - - - - - - - - - - - - - - ')
        print('모델 구축 완료')
        print('- - - - - - - - - - - - - - - - ')

    def predict_and_evaluate(self):
        print('- - - - - - - - - - - - - - - - ')
        print('최종 모델에 대한 성능 평가')
        print('- - - - - - - - - - - - - - - - ')


        prediction_result = predict_model(self.final_model,data = self.X_test)

        eval_Accuracy = check_metric(self.y_test, prediction_result['Label'], metric='Accuracy')
        print('Accuracy: ', eval_Accuracy)
        eval_f1 = check_metric(self.y_test, prediction_result['Label'] ,metric='F1')
        print('F1: ', eval_f1)
        eval_precision = check_metric(self.y_test, prediction_result['Label'], metric='Precision')
        print('Prec: ', eval_precision)
        eval_recall = check_metric(self.y_test, prediction_result['Label'], metric='Recall')
        print('Recall: ', eval_recall)

        return eval_Accuracy, eval_f1, eval_precision, eval_recall

    def model_visualiation(self):

        print('- - - - - - - - - - - - - - - - ')
        print('final 모델 성능평가 시각화')
        print('- - - - - - - - - - - - - - - - ')

        plot_model(estimator=self.final_model, plot='auc', save=True)
        plot_model(estimator=self.final_model, plot='pr', save=True)
        plot_model(estimator=self.final_model, plot='confusion_matrix',save=True)
        plot_model(estimator=self.final_model, plot='feature',save=True)


        # 모델 분석 후 각 플롯을 볼 수 있도록 사용자 인터페이스를 제공 ( 종합 )
        evaluate_model(estimator=self.final_model)

    def save_model(self):
        print('- - - - - - - - - - - - - - - - ')
        print('final 모델 저장')
        print('- - - - - - - - - - - - - - - - ')

        save_model(self.final_model, "final model")


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    data_preprocessor.load_raw_data()
    data_preprocessor.remove_duplicated()
    #data_preprocessor.remove_unnecessary_features()
    data_preprocessor.remove_outlier_based_std()


    data_preprocessor.make_scaled_data()
    data = data_preprocessor.put_cleaned_data()
    # data_preprocessor.put_cleaned_data_list()

    models = ModelMaker()
    models.load_data(data)
    models.prepare_labeled_data()
    models.split_data()
    models.prepare_model()
    models.predict_and_evaluate()
    models.save_model()










    

