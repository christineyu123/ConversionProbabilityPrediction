import os
import re
import string

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas_profiling import ProfileReport

from a_preprocessing.abstract_preprocessor import Preprocessor, PreprocessorSettings
from a_preprocessing.utils import split_timestamp_into_day_and_hour, check_no_missing_value, normalize_gaussian, \
    zip_lists_into_dict, \
    merge_and_divide


class ExplorePreprocessorSettings(PreprocessorSettings):
    output_folder = 'output_folder_explore_preprocessor'
    # Feature cols usable by models
    numerical_features_cols = ['startCount_gaussian',
                               'viewCount_gaussian',
                               'clickCount_gaussian',
                               'installCount_gaussian',
                               'startCount7d_gaussian',
                               'campaignId_install_over_frequency',
                               'sourceGameId_install_over_frequency']
    # Feature cols usable by models
    categorical_features_cols = ['timestamp_day_factorized',
                                 'timestamp_hour_factorized',
                                 'platform_factorized',
                                 'country_factorized',
                                 'connectionType_factorized']
    train_features_path = os.path.join(output_folder, 'train_features.csv')
    assessment_features_path = os.path.join(output_folder, 'assessment_features.csv')


class ExplorePreprocessor(Preprocessor):
    def __init__(self, train_data_path, assessment_data_path):
        super().__init__(train_data_path=train_data_path, assessment_data_path=assessment_data_path)

        self.is_visualize = False

        self.train_columns = []
        self.assessment_columns = []

        self.load_columns()

        print(f"\nVerify column: id\n")
        check_no_missing_value(self.raw_train_data, "id", "train", 'categorical')
        check_no_missing_value(self.raw_assessment_data, "id", "test", 'categorical')

    def load_columns(self):
        self.train_columns = self.raw_train_data.columns
        self.assessment_columns = self.raw_assessment_data.columns

        print(f"train columns: {self.train_columns}")
        print(f"test columns: {self.assessment_columns}")

    def categorical_variable_timestamp(self):
        return

    def analyze_timestamp(self, column):
        print(f"\nAnalyze column: {column}\n")
        check_no_missing_value(self.raw_train_data, column, "train", "categorical")
        check_no_missing_value(self.raw_assessment_data, column, "test", "categorical")

        self.raw_train_data = split_timestamp_into_day_and_hour(self.raw_train_data, column)
        self.raw_assessment_data = split_timestamp_into_day_and_hour(self.raw_assessment_data, column)
        self.encode_with_factorization(column1=f"{column}_day")
        self.encode_with_factorization(column1=f"{column}_hour")

        if self.is_visualize:
            self.draw_histograms(column1="day", column2="frequency", title="Day Distribution")
            self.draw_histograms(column1="hour", column2="frequency", title="Hour Distribution")

        self.raw_train_data = self.raw_train_data.drop([f"{column}_day", f"{column}_hour"], axis=1)
        self.raw_assessment_data = self.raw_assessment_data.drop([f"{column}_day", f"{column}_hour"], axis=1)
        return

    def analyze_categorical_variables(self, column, encode_methods):

        print(f"\nAnalyze column: {column}\n")
        check_no_missing_value(self.raw_train_data, column, "train", "categorical")
        check_no_missing_value(self.raw_assessment_data, column, "test", "categorical")

        if self.raw_train_data[column].dtype != object:
            self.raw_train_data[column] = self.raw_train_data[column].apply(lambda x: f"{str(x)}")
            self.raw_assessment_data[column] = self.raw_assessment_data[column].apply(lambda x: f"{str(x)}")

        if self.is_visualize:
            self.draw_histograms(column1=column, column2="frequency", title=f"{column} Frequency Distribution")

        if "install_over_frequency" in encode_methods:
            column_install_df = self.encode_with_install_over_frequency(column1=column)
            if self.is_visualize:
                self.draw_histograms_simple(column_install_df, column1=column, column2="install",
                                            title=f"{column} vs install")
                self.draw_histograms_simple(column_install_df, column1=column, column2="install_over_frequency",
                                            title=f"{column} vs {column}_install_over_frequency")

        if "factorization" in encode_methods:
            self.encode_with_factorization(column1=column)

        if "to_integer" in encode_methods:
            self.encode_with_to_first_integer(column1=column)

        if "factorization_over_letters" in encode_methods:
            def to_letter_only_lambda(my_text):
                my_text = re.sub('[0-9]+', '', my_text)
                my_text = my_text.translate(str.maketrans('', '', string.punctuation))
                return my_text

            self.raw_train_data[f'{column}_only_letters'] = self.raw_train_data[column].apply(to_letter_only_lambda)
            self.raw_assessment_data[f'{column}_only_letters'] = self.raw_assessment_data[column].apply(
                to_letter_only_lambda)
            self.encode_with_factorization(column1=f'{column}_only_letters')

            self.raw_train_data = self.raw_train_data.drop([f'{column}_only_letters'], axis=1)
            self.raw_assessment_data = self.raw_assessment_data.drop([f'{column}_only_letters'], axis=1)

    def analyze_continuous_variables(self, column):
        print(f"\nAnalyze column: {column}\n")
        check_no_missing_value(self.raw_train_data, column, "train", "numerical")
        check_no_missing_value(self.raw_assessment_data, column, "test", "numerical")
        if self.is_visualize:
            self.draw_histograms(column1=column, column2="frequency", title=f"{column} Frequency Distribution")

        # Apply gaussian
        self.raw_train_data, mean, variance = normalize_gaussian(self.raw_train_data, column)
        print(f"mean: {mean}, variance: {variance}")
        self.raw_assessment_data[f"{column}_gaussian"] = (self.raw_assessment_data[column] - mean) / variance

        # Apply log and gaussian
        res = np.log(self.raw_train_data[f'{column}'])
        res = np.where(np.isinf(res), 0, res)
        res = np.where(np.isneginf(res), 0, res)

        self.raw_train_data[f'{column}_log'] = res

        res = np.log(self.raw_assessment_data[f'{column}'])
        res = np.where(np.isinf(res), 0, res)
        res = np.where(np.isneginf(res), 0, res)

        self.raw_assessment_data[f'{column}_log'] = res

        self.raw_train_data, mean, variance = normalize_gaussian(self.raw_train_data, f'{column}_log')
        print(f"mean: {mean}, variance: {variance}")
        self.raw_assessment_data[f"{column}_log_gaussian"] = (self.raw_assessment_data[
                                                                  f'{column}_log'] - mean) / variance

    def draw_histograms(self, column1, column2=None, title=""):
        fig = go.Figure()

        if column2 == "install":
            fig.add_trace(go.Histogram(x=self.raw_train_data[column1], y=self.raw_train_data[column2], name='train'))

        elif column2 == "frequency":
            fig.add_trace(go.Histogram(x=self.raw_train_data[column1], name='train'))
            fig.add_trace(go.Histogram(x=self.raw_assessment_data[column1], name='test'))

        else:
            fig.add_trace(go.Histogram(x=self.raw_train_data[column1], y=self.raw_train_data[column2], name='train'))
            fig.add_trace(
                go.Histogram(x=self.raw_assessment_data[column1], y=self.raw_assessment_data[column2], name='test'))

        # Overlay both histograms
        fig.update_layout(
            title_text=title,
            xaxis_title_text=column1,
            yaxis_title_text=column2,
            bargap=0.2,
            bargroupgap=0.1
        )

        fig.show()

    def draw_histograms_simple(self, df, column1, column2, title):
        fig = px.histogram(df, x=df.index, y=column2, title=title)
        fig.show()

    def encode_with_install_over_frequency(self, column1, column2='install'):
        column1_frequency_dict = self.raw_train_data[column1].value_counts(sort=False).to_dict()

        temp_df = self.raw_train_data[[column1, column2]].set_index(column1)
        temp_df = temp_df.groupby(temp_df.index).sum()

        column1_install_dict = temp_df.to_dict()[column2]
        column1_install_frequency_dict = merge_and_divide(column1_install_dict, column1_frequency_dict)

        self.raw_train_data[f'{column1}_{column2}_over_frequency'] = self.raw_train_data[column1].apply(
            lambda x: column1_install_frequency_dict[x])
        self.raw_assessment_data[f'{column1}_{column2}_over_frequency'] = self.raw_assessment_data[column1].apply(
            lambda x: column1_install_frequency_dict.get(x, 0))

        temp_df[f'{column2}_over_frequency'] = temp_df.index
        temp_df[f'{column2}_over_frequency'] = temp_df[f'{column2}_over_frequency'].apply(
            lambda x: column1_install_frequency_dict[x])

        return temp_df

    def encode_with_factorization(self, column1):
        codes, unique = pd.factorize(self.raw_train_data[column1])
        self.raw_train_data[f"{column1}_factorized"] = codes

        codes = list(range(unique.size))
        unique = list(unique)
        print(f"Total unique values: {len(unique)}, {unique}")

        codes_dict, unseen_index = zip_lists_into_dict(unique, codes)
        self.raw_assessment_data[f"{column1}_factorized"] = self.raw_assessment_data[column1].apply(
            lambda x: codes_dict.get(x, unseen_index))

    def encode_with_to_first_integer(self, column1):
        def to_int_lambda(s):
            first_char = str(s).split('.')[0]
            try:
                final_int = int(first_char)
            except:
                final_int = 0
            return final_int

        self.raw_train_data[f'{column1}_to_integer'] = self.raw_train_data[column1].apply(to_int_lambda)
        self.raw_assessment_data[f'{column1}_to_integer'] = self.raw_assessment_data[column1].apply(to_int_lambda)
        return

    def plot_pos_neg_assess_profiles(self, cols):
        pos = self.raw_train_data[self.raw_train_data['install'] == 1]
        neg = self.raw_train_data[self.raw_train_data['install'] == 0]
        test = self.raw_assessment_data

        pos_profile = ProfileReport(pos)
        neg_profile = ProfileReport(neg)
        test_profile = ProfileReport(test)

        pos[cols].describe().to_csv("./pos_numerical_profiles.csv", index=False)
        neg[cols].describe().to_csv("./neg_numerical_profiles.csv", index=False)
        test[cols].describe().to_csv("./test_numerical_profiles.csv", index=False)

        pos_profile.to_file("./pos_profile.html")
        neg_profile.to_file("./neg_profile.html")
        test_profile.to_file("./test_profile.html")
        return

    def generate_features(self):
        if self.is_visualize:
            numerical_cols = ['startCount',
                              'viewCount',
                              'clickCount',
                              'installCount',
                              'startCount1d',
                              'startCount7d'
                              ]
            self.plot_pos_neg_assess_profiles(cols=numerical_cols)
        self.analyze_timestamp(column="timestamp")
        self.analyze_categorical_variables("campaignId", encode_methods=["install_over_frequency", "factorization"])
        self.analyze_categorical_variables("platform", encode_methods=["factorization"])
        self.analyze_categorical_variables("softwareVersion", encode_methods=["to_integer"])
        self.analyze_categorical_variables("sourceGameId", encode_methods=["install_over_frequency", "factorization"])
        self.analyze_categorical_variables("country", encode_methods=["factorization"])
        self.analyze_continuous_variables("startCount")
        self.analyze_continuous_variables("viewCount")
        self.analyze_continuous_variables("clickCount")
        self.analyze_continuous_variables("installCount")
        self.analyze_timestamp(column='lastStart')
        self.analyze_continuous_variables("startCount1d")
        self.analyze_continuous_variables("startCount7d")
        self.analyze_categorical_variables("connectionType", encode_methods=['factorization'])
        self.analyze_categorical_variables("deviceType", encode_methods=["factorization_over_letters"])

        if not os.path.exists(ExplorePreprocessorSettings.output_folder):
            os.makedirs(ExplorePreprocessorSettings.output_folder)

        self.raw_train_data.to_csv(ExplorePreprocessorSettings.train_features_path, sep=';', index=False)

        self.raw_assessment_data.to_csv(ExplorePreprocessorSettings.assessment_features_path, sep=';', index=False)
