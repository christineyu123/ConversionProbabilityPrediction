import os

from sklearn import preprocessing

from a_preprocessing.abstract_preprocessor import Preprocessor, PreprocessorSettings


class NaivePreprocessorSettings(PreprocessorSettings):
    output_folder = 'output_folder_naive_preprocessor'
    # Feature cols usable by models
    numerical_features_cols = ['startCount',
                               'viewCount',
                               'clickCount',
                               'installCount',
                               'startCount1d',
                               'startCount7d']
    # Feature cols usable by models
    categorical_features_cols = ['sourceGameId',
                                 'campaignId',
                                 'platform',
                                 'softwareVersion',
                                 'country',
                                 'connectionType',
                                 'deviceType']
    train_features_path = os.path.join(output_folder, 'train_features.csv')
    assessment_features_path = os.path.join(output_folder, 'assessment_features.csv')


class NaivePreprocessor(Preprocessor):
    def __init__(self, train_data_path, assessment_data_path):
        super().__init__(train_data_path=train_data_path, assessment_data_path=assessment_data_path)

    def scale_numerical_features(self, df, col_names, existing_standardScaler):
        if existing_standardScaler is not None:
            standardScaler = existing_standardScaler
            df[col_names] = standardScaler.transform(df[col_names])
        else:
            standardScaler = preprocessing.StandardScaler()
            df[col_names] = standardScaler.fit_transform(df[col_names])
        return df, standardScaler

    def encode_categorical_features(self, df, col_names, existing_col_to_cat_mapping):
        if existing_col_to_cat_mapping is not None:
            col_to_cat_mapping = existing_col_to_cat_mapping
            col_to_unknown_index = {col: len(values) for col, values in col_to_cat_mapping.items()}
            for col, mapping in col_to_cat_mapping.items():
                df[col] = df[col].map(lambda x: mapping.get(x, col_to_unknown_index[col]))
        else:
            col_to_cat_mapping = {
                col: dict(zip(values, range(len(values))))
                for col, values in map(lambda col: (col, df[col].unique()), col_names)
            }
            for col, mapping in col_to_cat_mapping.items():
                df[col] = df[col].map(mapping.get)
        return df, col_to_cat_mapping

    def generate_features(self):
        # Input columns to consider
        numerical_features_cols = ['startCount',
                                   'viewCount',
                                   'clickCount',
                                   'installCount',
                                   'startCount1d',
                                   'startCount7d']
        categorical_features_cols = ['sourceGameId',
                                     'campaignId',
                                     'platform',
                                     'softwareVersion',
                                     'country',
                                     'connectionType',
                                     'deviceType']

        all_feature_cols = numerical_features_cols + categorical_features_cols + ['install']

        trainDF = self.raw_train_data[all_feature_cols]

        # Fill na
        trainDF[categorical_features_cols] = trainDF[categorical_features_cols].astype(str).fillna('')
        trainDF[numerical_features_cols] = trainDF[numerical_features_cols].astype(float).fillna(0.0)

        # Process numerical features
        trainDF, standardScaler = self.scale_numerical_features(df=trainDF,
                                                                col_names=numerical_features_cols,
                                                                existing_standardScaler=None)

        # Process categorical features
        trainDF, col_to_cat_mapping = self.encode_categorical_features(df=trainDF,
                                                                       col_names=categorical_features_cols,
                                                                       existing_col_to_cat_mapping=None)

        # process assessment data
        assessmentDF = self.raw_assessment_data[numerical_features_cols + categorical_features_cols + ['id']]

        # Fill na
        assessmentDF[categorical_features_cols] = assessmentDF[categorical_features_cols].astype(str).fillna('')
        assessmentDF[numerical_features_cols] = assessmentDF[numerical_features_cols].astype(float).fillna(0.0)

        # Process numerical features
        assessmentDF, _ = self.scale_numerical_features(df=assessmentDF,
                                                        col_names=numerical_features_cols,
                                                        existing_standardScaler=standardScaler)

        # Process categorical features
        assessmentDF, _ = self.encode_categorical_features(df=assessmentDF,
                                                           col_names=categorical_features_cols,
                                                           existing_col_to_cat_mapping=col_to_cat_mapping)

        if not os.path.exists(NaivePreprocessorSettings.output_folder):
            os.makedirs(NaivePreprocessorSettings.output_folder)

        trainDF.to_csv(NaivePreprocessorSettings.train_features_path, sep=';', index=False)

        assessmentDF.to_csv(NaivePreprocessorSettings.assessment_features_path, sep=';', index=False)
