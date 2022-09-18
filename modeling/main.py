from data_exploration import DataAnalyzer
from data_transform import DataPreprocessor
from model_pycaret import ModelMaker

if __name__ == "__main__":
    data_analyzer = DataAnalyzer()
    data_analyzer.show_statistics()
    data_analyzer.show_label_info()
    data_analyzer.show_correlation()
    data_analyzer.show_correlation_label_attribute()
    data_analyzer.check_normality_test()

    data_preprocessor = DataPreprocessor()
    data_preprocessor.load_raw_data()
    data_preprocessor.remove_unnecessary_features()
    data_preprocessor.remove_incorrect_data()
    data_preprocessor.address_missing_value()

    data_preprocessor.make_scaled_data()
    data = data_preprocessor.put_cleaned_data()


    models = ModelMaker()
    models.load_data()
    models.split_data()
    models.prepare_model()
    models.predict_and_evaluate()
    models.save_model()
