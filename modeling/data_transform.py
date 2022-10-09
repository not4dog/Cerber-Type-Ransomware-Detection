import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer


class DataPreprocessor:
    def __init__(self):
        self.raw_data = None
        self.pd_data = None

        self.pd_scaled_data = None

        self.pd_scaled_data_list = []
        self.load_raw_data()

    def load_raw_data(self):
        self.raw_data = pd.read_csv("EndAnalysis.csv")

    def remove_duplicated(self):
        # SHA-256 기준으로 중복 값 제거
        dups = (self.raw_data.duplicated('SHA-256'))
        print('Number of duplicate  = %d' % (dups.sum()))

        self.pd_data = self.raw_data.drop_duplicates(['SHA-256'])

    # 불필요한 특징 제거 ( SHA-256, 상관관계로 발견한 의미 없는 feature 제거
    def remove_unnecessary_features(self):
        self.pd_data = self.raw_data.drop(['SHA-256'], axis = 1)

    def check_missing_col(self,dataframe):
        missing_col = []
        for col in dataframe.columns:
            missing_values = sum(dataframe[col].isna())
            is_missing = True \
                if missing_values >= 1 else False
            if is_missing:
                print(f'결측치가 있는 컬럼은: {col} 입니다')
                print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
                missing_col.append([col, dataframe[col].dtype])
        if missing_col == []:
            print('결측치가 존재하지 않습니다')
        return missing_col

    def handle_na(data, missing_col):
        temp = data.copy()
        for col, dtype in missing_col:
            if dtype == 'O':
                # 범주형 feature가 결측치인 경우 해당 행들을 삭제
                temp = temp.dropna(subset=[col])
        return temp

    def address_missing_value(self):
        # missing value(결측값:0)를 중간값으로 imputation(대체) 하기 - 사용 X (정상파일 같은 경우 동적분석 0인 부분이 나오므로 사용 X )
        imputer = SimpleImputer(missing_values=0, strategy="median")
        check_imputer = imputer.fit_transform(self.pd_data)

        print('결측치 중간값으로 대체: ', check_imputer)

    def remove_outlier_based_std(self):
        # Z-score 기반 이상치 제거 ( Z-score가 3보다 크면 이상값 ) - 제거대신 스케일링으로 대체 ( 제거하니까 데이터 대부분이 사라지는 현상 발생 )
        for i in range(0, len(self.pd_data.iloc[1])):
            # self.pd_data.iloc[:, i] = self.pd_data.iloc[:, i].replace(0, np.NaN)  # NaN을 0으로 처리 (선택적)
            self.pd_data = self.pd_data[~(np.abs(self.pd_data.iloc[:, i] - self.pd_data.iloc[:, i].mean()) > (3 * self.pd_data.iloc[:, i].std()))].fillna(0)

        print(self.pd_data.shape)

    def remove_outlier_based_IQR(self):
        # IQR(사분위수)를 이용한 이상치 제거 - 제거대신 스케일링으로 대체 ( 제거하니까 데이터 대부분이 사라지는 현상 발생 )
        quartile_1, quartile_3 = np.percentile(self.pd_data, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        self.pd_data = np.where((self.pd_data > upper_bound) | (self.pd_data < lower_bound))

        print(self.pd_data)

    def make_scaled_data_list(self):
        minmax = MinMaxScaler()
        maxabs = MaxAbsScaler()
        standard = StandardScaler()
        robust = RobustScaler()
        normal = Normalizer()

        minmax.fit(self.pd_data)
        maxabs.fit(self.pd_data)
        standard.fit(self.pd_data)
        robust.fit(self.pd_data)
        normal.fit(self.pd_data)


        self.pd_scaled_data_list.append(minmax.transform(self.pd_data))
        self.pd_scaled_data_list.append(maxabs.transform(self.pd.data))
        self.pd_scaled_data_list.append(standard.transform(self.pd_data))
        self.pd_scaled_data_list.append(robust.transform(self.pd_data))
        self.pd_scaled_data_list.append(normal.transform(self.pd_data))

    def make_scaled_data(self):

        # 정규화 -  서로 다른 피처의 크기를 통일하기 위해 크기를 변환
        scaler = RobustScaler()
        scaler.fit(self.pd_data)

        self.pd_scaled_data = scaler.transform(self.pd_data)

    def put_cleaned_data(self):
        print('[스케일링 된 데이터 반환]')
        print(self.pd_scaled_data)
        print()

        return self.pd_scaled_data

    def put_cleaned_data_list(self):
        print('[다양한 방식으로 re-scaling 된 데이터 리스트 반환]')
        print(self.pd_scaled_data_list)
        print()

        return self.pd_scaled_data_list

# 모델 성능 안 좋을 시 특징 선택법을 이용
# 모델을 돌릴 때 쓸모 없는 변수들을 제거함으로써모델의 속도 개선, 오버피팅 방지 등의 효과를 얻기 위해 사용하는 방법.
# 1. Wrapper method : 모델링 돌리면서 변수 채택(주로 사용) && 2. Filter Method : 전처리단에서 통계기법 사용하여 변수 채택 &&3. Embedded method : 라쏘, 릿지, 엘라스틱넷 등 내장함수 사용하여 변수 채택
# 출처: https://dyddl1993.tistory.com/18 [새우위키:티스토리]

# class FeatureEngineer:
#     def __init__(self):
#         self.pd_data = None  # 정제중인 데이터
#         self.load_raw_data()
#     def load_raw_data(self):
#         self.raw_data = pd.read_csv("All_Feature_CTRD_Data.csv")

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    data_preprocessor.load_raw_data()
    data_preprocessor.remove_duplicated()
    data_preprocessor.remove_unnecessary_features()
    #data_preprocessor.address_missing_value()
    #data_preprocessor.remove_outlier_based_std()
    #data_preprocessor.remove_outlier_based_IQR()


    # data_preprocessor.put_cleaned_data_list()



