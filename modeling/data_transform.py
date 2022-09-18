import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


class DataPreprocessor:
    def __init__(self):
        self.raw_data = None
        self.pd_data = None
        self.pd_scaled_data = None

        self.pd_scaled_data_list = []
        self.load_raw_data()

    def load_raw_data(self):
        bengin = pd.read_csv("bengin_frequency.csv")
        cerber = pd.read_csv("cerber_frequency.csv")
        df = pd.concat([bengin, cerber])
        self.raw_data = df


    # 불필요한 특징 제거
    def remove_unnecessary_features(self):
        self.pd_data = self.raw_data.drop(['SHA-256'], axis = 1)

    def remove_incorrect_data(self):
        # 1) 레이블이 NaN을 포함한 row 제외
        check_NaN = self.pd_data['family'].isnull().any()

        if check_NaN:
            print('레이블 값에 NaN가 존재하는 row 제거')
            self.pd_data = self.pd_data.dropna(subset=["family"])

        print('[NaN data들 제거]')
        # print(self.pd_data)
        print()

    def address_missing_value(self):
        # 2) missing value(결측값)를 중간값으로 imputation(대체) 하기
        imputer = SimpleImputer(missing_values=np.NaN, strategy="median")
        check_imputer = imputer.fit_transform(self.pd_data)

        # print('결측치 중간값으로 대체: ', check_imputer)

    def remove_outlier_based_std(self):
        # 표준점수 기반 이상치 제거
        for i in range(0, len(self.pd_data.iloc[1])):
            self.pd_data.iloc[:, i] = self.pd_data.iloc[:, i].replace(0, np.NaN)  # 0을 NaN처리
            self.pd_data = self.pd_data[~(np.abs(self.pd_data.iloc[:, i] - self.pd_data.iloc[:, i].mean()) > (3 * self.pd_data.iloc[:, i].std()))].fillna(0)

        print()
    def remove_outlier_based_IQR(self):
        # IQR를 이용한 이상치 제거
        quartile_1, quartile_3 = np.percentile(self.pd_data, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        self.pd_data =  np.where((self.pd_data > upper_bound) | (self.pd_data < lower_bound))

    def make_scaled_data_list(self):
        minmax = MinMaxScaler()
        maxabs = MaxAbsScaler()
        standard = StandardScaler()
        robust = RobustScaler()

        minmax.fit(self.pd_data)
        maxabs.fit(self.pd_data)
        standard.fit(self.pd_data)
        robust.fit(self.pd_data)

        self.pd_scaled_data_list.append(minmax.transform(self.pd_data))
        self.pd_scaled_data_list.append(maxabs.transform(self.pd.data))
        self.pd_scaled_data_list.append(standard.transform(self.pd_data))
        self.pd_scaled_data_list.append(robust.transform(self.pd_data))

    def make_scaled_data(self):
        standard = StandardScaler()
        standard.fit(self.pd_data)

        self.pd_scaled_data = standard.transform(self.pd_data)

    def put_cleaned_data(self):
        print('[스케일링 된 데이터 반환]')
        print(self.pd_scaled_data)
        print()

        return self.pd_scaled_data

    def put_cleaned_data_list(self):
        print('[다양한 방식으로 re-scaling 된 데이터 리스트 반환]')
        print(self.pd_scaled_data_list)
        return self.pd_scaled_data_list


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    data_preprocessor.load_raw_data()
    data_preprocessor.remove_unnecessary_features()
    data_preprocessor.remove_incorrect_data()
    data_preprocessor.address_missing_value()

    data_preprocessor.make_scaled_data()
    data = data_preprocessor.put_cleaned_data()
    data_preprocessor.put_cleaned_data_list()



