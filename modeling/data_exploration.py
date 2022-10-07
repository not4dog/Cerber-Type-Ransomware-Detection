import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from scipy.stats import skew, kurtosis


from scipy.stats import shapiro
from scipy.stats import chi2_contingency

# 데이터 분석
class DataAnalyzer:
    def __init__(self):
        self.raw_data = pd.read_csv("All_Feature_CTRD_Data.csv")

        print('[raw data 정보]')
        print(self.raw_data)
        print()

        print('[raw data 칼럼 정보]')
        print(self.raw_data.info())
        print()

        # 의미없는 속성 제거한 raw 데이터
        self.meaningful_raw_data = None


    def show_statistics(self):
        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        print('[raw data 통계 정보]')
        print(self.raw_data.describe())
        print()

    def show_label_info(self):
        print('[정상과 Cerber 구별 정보]')
        print(self.raw_data['Cerber'].value_counts())
        print()

    def show_correlation(self):
        self.meaningful_raw_data = self.raw_data.drop(['SHA-256',"Cerber"], axis = 1)

        print('[컬럼별 상관관계 정보]')
        print(self.meaningful_raw_data.corr())
        print()

    def show_correlation_visual(self):
        corr = self.meaningful_raw_data.corr()

        # 사이즈 지정
        fig, ax = plt.subplots(figsize=(10, 10))
        # 히트맵
        sns.heatmap(corr,
                    cmap='RdYlBu_r',
                    annot=True,  # 실제 값을 표시한다
                    linewidths=.5,  # 경계면 실선으로 구분하기
                    cbar_kws={"shrink": .5},  # 컬러바 크기 절반으로 줄이기
                    vmin=-1, vmax=1  # 컬러바 범위 -1 ~ 1
                    )
        plt.title("Dataset Correlation_Visual")
        # 그래프를 이미지 파일 등으로 저장
        plt.savefig('Dataset Correlation')
        plt.show()

    def show_correlation_label_attribute(self):
        print('[레이블과 다른 속성들 사이의 상관관계]')
        corr_matrix = self.raw_data.corr()
        print(corr_matrix["Cerber"].sort_values(ascending=False))
        print()

    def check_normality_test(self):
        # 데이터셋이 어떤식으로 정규분포가 이뤄졌는지 확인하여 스케일링을 결정
        # 표본수(n)이므,로 2000미만 데이터셋에 적합한 정규성 검정을 위해 Shaprio-Wiliks
        # 귀무가설 H0 : 데이터셋이 정규분포를 따른다
        # 대립가설 H1 : 데이터셋이 정규분포를 따르지 X
        
        print('[feature 별 정규성 검정]')
        for col, item in self.meaningful_raw_data.iteritems():
            print()
            print(shapiro(self.meaningful_raw_data[col]) )

        print()
        # 결과 : P값이 0.05(일반적인 귀무가설 검정 임계치 수준)보다 높은 부분이 보이므로 모든 feature가 정규분포를 따르지 않음

# 시각화
class DataVisualizer:
    def __init__(self):
        self.load_raw_data()

    def load_raw_data(self):
        self.raw_data = pd.read_csv("All_Feature_CTRD_Data.csv")
        self.meaningful_raw_data = self.raw_data.drop(['SHA-256',"Cerber"], axis = 1)

    def show_box_plot(self):
        sns.boxplot(orient="h",data=self.meaningful_raw_data)
        plt.title("Dataset Box Plot")
        plt.savefig('Dataset_Box Plot')
        plt.show()

    def show_bar_plot(self):
        sns.barplot(self.meaningful_raw_data)
        plt.title("Dataset Bar Plot")
        plt.savefig('Dataset_Bar Plot')
        plt.show()

    def show_scatter_plot(self):
        sns.scatterplot(self.meaningful_raw_data)
        plt.title("Dataset Scatter Plot")
        plt.savefig('Dataset_Scatter Plot')
        plt.show()

    def show_dist_plot(self):
        sns.distplot(self.meaningful_raw_data)
        plt.title("Dataset distribution Plot")
        plt.savefig('Dataset_distribution Plot')
        plt.show()



if __name__ == '__main__':
    data_analyzer = DataAnalyzer()
    data_analyzer.show_statistics()
    data_analyzer.show_label_info()
    data_analyzer.show_correlation()
    data_analyzer.show_correlation_visual()
    data_analyzer.show_correlation_label_attribute()
    data_analyzer.check_normality_test()

    data_visualizer = DataVisualizer()
    data_visualizer.show_dist_plot()
    data_visualizer.show_box_plot()
    # data_visualizer.show_bar_plot()
    # data_visualizer.show_scatter_plot()
    # data_visualizer.show_QQ_plot()
