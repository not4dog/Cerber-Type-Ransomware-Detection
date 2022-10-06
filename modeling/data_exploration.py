import pandas as pd
import numpy as np
from scipy.stats import kstest

import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 분석
class DataAnalyzer:
    def __init__(self):
        self.load_raw_data()
        print('[raw data 정보]')
        print(self.raw_data)
        print()

        print('[raw data 칼럼 정보]')
        print(self.raw_data.info())
        print()

        # 의미없는 속성 제거한 raw 데이터
        self.meaningful_raw_data = None

    def load_raw_data(self):
        bengin = pd.read_csv("bengin_frequency.csv")
        cerber = pd.read_csv("cerber_frequency.csv")
        df = pd.concat([bengin, cerber])
        self.raw_data = df


    def show_statistics(self):
        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        print('[raw data 통계 정보]')
        print(self.raw_data.describe())
        print()

    def show_label_info(self):
        print('[레이블 정보]')
        print(self.raw_data['family'].value_counts())
        print()

    def show_correlation(self):
        self.meaningful_raw_data = self.raw_data.drop(['SHA-256'], axis = 1)

        print('[컬럼별 상관관계 정보]')
        print(self.meaningful_raw_data.corr())
        print()

    def show_correlation_visual(self):
        df = self.meaningful_raw_data.corr()

        # 사이즈 지정
        fig, ax = plt.subplots(figsize=(7, 7))

        # 히트맵
        sns.heatmap(df,
                    cmap='RdYlBu_r',
                    annot=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .5},
                    vmin=-1, vmax=1
                    )
        plt.title("Dataset Correlation_Visual")
        # 그래프를 이미지 파일 등으로 저장
        plt.savefig('Dataset Correlation')
        plt.show()

    def show_correlation_label_attribute(self):
        print('[레이블과 다른 속성들 사이의 상관관계]')
        corr_matrix = self.raw_data.corr()
        print(corr_matrix["family"].sort_values(ascending=False))
        print()

    def check_normality_test(self):

        print('[feature 별 정규성 검정]')
        for col, item in self.meaningful_raw_data.iteritems():
            print(kstest(self.meaningful_raw_data[col],'norm') )

        print()

# 시각화
class DataVisualizer:
    def __init__(self):
        self.load_raw_data()

    def load_raw_data(self):
        bengin = pd.read_csv("bengin_frequency.csv")
        cerber = pd.read_csv("cerber_frequency.csv")
        df = pd.concat([bengin, cerber])
        self.raw_data = df
        self.meaningful_raw_data = self.raw_data.drop(['SHA-256'], axis=1)

    def show_dist_plot(self):
        sns.distplot(self.meaningful_raw_data)
        plt.title("Dataset distribution Plot")
        plt.savefig('Dataset_distribution Plot')
        plt.show()

    def show_box_plot(self):
        sns.boxplot(self.meaningful_raw_data)
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

if __name__ == '__main__':
    data_analyzer = DataAnalyzer()
    data_analyzer.show_statistics()
    data_analyzer.show_label_info()
    data_analyzer.show_correlation()
    data_analyzer.show_correlation_visual()
    data_analyzer.show_correlation_label_attribute()
    data_analyzer.check_normality_test()

    # data_visualizer = DataVisualizer()
    # # data_visualizer.show_dist_plot()
    # data_visualizer.show_box_plot()
    # data_visualizer.show_bar_plot()
    # data_visualizer.show_scatter_plot()
