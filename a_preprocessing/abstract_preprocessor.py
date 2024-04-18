from abc import ABC, abstractmethod

import pandas as pd


class PreprocessorSettings:
    output_folder = ''
    numerical_features_cols = []
    categorical_features_cols = []
    train_features_path = ''
    assessment_features_path = ''


class Preprocessor(ABC):

    def __init__(self, train_data_path, assessment_data_path):
        self.raw_train_data = pd.read_csv(train_data_path, delimiter=';')
        self.raw_assessment_data = pd.read_csv(assessment_data_path, delimiter=';')

    @abstractmethod
    def generate_features(self):
        pass
