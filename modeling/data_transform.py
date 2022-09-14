import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

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
        print(self.pd_data)
        print()

    # def remove_no_frequency(self):
    # # 2) 빈도수가 0인 부분 제외
    #     mask = self.pd_data['push' or 'mov' or 'call' or 'sub' or 'jmp' or 'add' or 'cmp' or 'test' or 'lea' or 'pop' or 'FindFirstFile' or 'SearchPathW' or 'SetFilePointer' or 'FindResourceEx' or 'GetFileAttributesW' or 'SetFileAttributesW' or 'SetFilePointerEx' or 'CryptEncrypt' or 'CreateThread' or 'FindResourceExW'].isin([0])
    #     if mask:
    #         print("빈도수가 0인 부분 제외")
    #         self.pd_data = self.pd_data.dropna(subset=["family"])
    #
    #     print('[빈도수가 0인 부분 제외]')
    #     print(self.pd_data)
    #     print()

    def address_missing_value(self):
        # 3) missing value(결측값)에 대해 imputation(대체) 하기
        imputer = SimpleImputer() # 어떻게 imputation 할지는 데이터 특성에 따라 갈린다고 함 ( 기본값 : 평균 설정 )
        # 예를 들어, 시계열데이터의 경우, interpolation으로 impute해서, 결측치 처리함

        check_NaN = self.pd_data.isnull().any().any()
        print('결측치가 존재여부: ', check_NaN)

    def show_data_profile(self):
        pr = self.pd_data.profile_report()
        print("데이터 프로필 : ", pr)

    def remove_outlier_based_std(self):
        for i in range(0, len(self.pd_data.iloc[1])):
            self.pd_data.iloc[:, i] = self.pd_data.iloc[:, i].replace(0, np.NaN)  # optional
            self.pd_data = self.pd_data[~(np.abs(self.pd_data.iloc[:, i] - self.pd_data.iloc[:, i].mean()) > (3 * self.pd_data.iloc[:, i].std()))].fillna(0)

    def remove_outlier_based_IQR(self):
        quartile_1, quartile_3 = np.percentile(self.pd_data, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        self.pd_data =  np.where((self.pd_data > upper_bound) | (self.pd_data < lower_bound))

    def make_scaled_data_list(self):
        minmax = MinMaxScaler()
        standard = StandardScaler()
        robust = RobustScaler()
        # quantile = QuantileTransformer()
        # power = PowerTransformer()

        minmax.fit(self.pd_data)
        standard.fit(self.pd_data)
        robust.fit(self.pd_data)
        # quantile.fit(self.pd_data)
        # power.fit(self.pd_data)

        self.pd_scaled_data_list.append(minmax.transform(self.pd_data) )
        self.pd_scaled_data_list.append(standard.transform(self.pd_data))
        self.pd_scaled_data_list.append(robust.transform(self.pd_data))
        # self.pd_scaled_data_list.append(quantile.transform(self.pd_data))
        # self.pd_scaled_data_list.append(power.transform(self.pd_data))

    def put_cleaned_data_list(self):
        print('[다양한 방식으로 re-scaling 된 데이터 리스트 반환]')
        print()
        return self.pd_scaled_data_list

    def make_scaled_data(self):
        standard = StandardScaler()
        standard.fit(self.pd_data)

        self.pd_scaled_data = standard.transform(self.pd_data)

    def put_cleaned_data(self):
        return self.pd_scaled_data

if __name__ == '__main__':
    data_preprocessor = DataPreprocessor()
    data_preprocessor.load_raw_data()
    data_preprocessor.remove_unnecessary_features()
    data_preprocessor.remove_incorrect_data()
    #data_preprocessor.remove_no_frequency()
    data_preprocessor.address_missing_value()


    data_preprocessor.make_scaled_data_list()
    data_list = data_preprocessor.put_cleaned_data_list()
    data_preprocessor.make_scaled_data()
    data = data_preprocessor.put_cleaned_data()
    print(data)