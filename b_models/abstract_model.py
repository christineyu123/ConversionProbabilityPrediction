from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from a_preprocessing.abstract_preprocessor import PreprocessorSettings


class Model(ABC):

    def __init__(self, preprocessing_settings: PreprocessorSettings, model_name, visualize=False):
        self.preprocessing_settings = preprocessing_settings
        self.numerical_features_cols = preprocessing_settings.numerical_features_cols
        self.categorical_features_cols = preprocessing_settings.categorical_features_cols
        self.all_cols_and_install = self.numerical_features_cols + self.categorical_features_cols + ['install']
        self.all_cols_and_id = self.numerical_features_cols + self.categorical_features_cols + ['id']
        train_feature_df = pd.read_csv(self.preprocessing_settings.train_features_path, delimiter=';')[
            self.all_cols_and_install]
        self.emb_counts = [max(len(train_feature_df[c].unique()), train_feature_df[c].max() + 1) + 1 for c in
                           self.categorical_features_cols]
        self.batch_size = 128
        self.model_name = model_name
        self.visualize = visualize
        self.training_df = None
        self.valid_df = None
        self.test_df = None

    def create_train_val_test_df(self):
        # Get a random 60% of the entire set for training
        train_feature_df = pd.read_csv(self.preprocessing_settings.train_features_path, delimiter=';')[
            self.all_cols_and_install]
        self.training_df, valtest_df = train_test_split(train_feature_df,
                                                        test_size=0.4,
                                                        stratify=train_feature_df['install'].values,
                                                        random_state=1)
        self.valid_df, self.test_df = train_test_split(valtest_df,
                                                       test_size=0.5,
                                                       stratify=valtest_df['install'].values,
                                                       random_state=1)

    def resample_dataset(self, df):
        features = np.array(df)
        labels = np.array(df['install'])
        bool_labels = labels != 0
        pos_features = features[bool_labels]
        neg_features = features[~bool_labels]

        ids = np.arange(len(pos_features))
        choices = np.random.choice(ids, len(neg_features))

        res_pos_features = pos_features[choices]

        resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
        resampled_df = pd.DataFrame(resampled_features, columns=df.columns)
        resampled_df = resampled_df.sample(frac=1)  # Shuffle the dataframe
        return resampled_df

    @abstractmethod
    def clean_variables(self):
        pass

    @abstractmethod
    def split_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval_on_valid(self):
        pass

    @abstractmethod
    def eval_on_test(self):
        pass

    @abstractmethod
    def generate_assessment_prediction(self):
        pass
