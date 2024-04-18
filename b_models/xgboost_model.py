import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from lightgbm import plot_metric
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBClassifier
from xgboost import plot_importance

from a_preprocessing.abstract_preprocessor import PreprocessorSettings
from b_models.abstract_model import Model
from metrics.metric import get_all_metrics, return_counts_y_pred

gpu_list = tf.config.list_physical_devices('GPU')
tree_method = 'gpu_hist' if len(gpu_list) > 0 else None


class XGBoostModel(Model):

    def __init__(self, preprocessing_settings: PreprocessorSettings, model_name, is_resample):
        super().__init__(preprocessing_settings=preprocessing_settings, model_name=model_name)

        self.output_folder = 'output_folder_XGBoostModel'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.best_model_output_path = os.path.join(self.output_folder, self.model_name + '.pkl')

        self.is_resample = is_resample

    def get_model(self):
        model = XGBClassifier(
            objective='binary:logistic',
            tree_method=tree_method,
            n_jobs=-1,
            n_estimators=1000,
            max_depth=16,
            colsample_bytree=0.8,
            subsample=0.8,
            learning_rate=0.2,
            min_child_weight=6
        )
        return model

    def split_data(self):
        # Split into 3 subset df
        self.create_train_val_test_df()

        if self.is_resample:
            self.training_df = self.resample_dataset(df=self.training_df)

        self.training_set_y = self.training_df[['install']]
        self.training_set_x = self.training_df.drop(['install'], axis=1)
        self.valid_set_y = self.valid_df[['install']]
        self.valid_set_x = self.valid_df.drop(['install'], axis=1)
        self.test_set_y = self.test_df[['install']]
        self.test_set_x = self.test_df.drop(['install'], axis=1)

    def train(self):
        self.split_data()
        model = self.get_model()
        model.fit(
            self.training_set_x,
            self.training_set_y,
            eval_metric='logloss',
            eval_set=[(self.training_set_x, self.training_set_y), (self.valid_set_x, self.valid_set_y)],
            verbose=True,
            early_stopping_rounds=10
        )

        if self.visualize:
            plt.rcParams['figure.facecolor'] = 'white'
            evals_result = model.evals_result()
            ax = plot_metric(evals_result, metric='logloss')
            plt.title(f'Xgboost Learning Curve for Model: {self.model_name}')
            plt.show()

            fig, ax = plt.subplots(figsize=(7, 10))
            plot_importance(model,
                            ax=ax,
                            height=0.5).set(xlabel='feature importance',
                                            title=f'Feature importance for Model: {self.model_name}',
                                            ylabel='feature')

        # save
        pickle.dump(model, open(self.best_model_output_path, "wb"))

        # save top N important feature
        thresholds = list(model.feature_importances_)
        feature_names = list(self.training_set_x.columns)
        feature_to_importance_map = {f: t for f, t in zip(feature_names, thresholds)}
        top_N_features = sorted(feature_to_importance_map.items(), key=lambda x: x[1], reverse=True)[:10]
        top_N_features = [f_and_t[0] for f_and_t in top_N_features]
        top_N_features_path = os.path.join(self.output_folder, f'{self.model_name}_top_N_features.json')
        json.dump(top_N_features, open(top_N_features_path, "w"))

    def exec_eval(self, set_x, set_y):

        model = pickle.load(open(self.best_model_output_path, "rb"))
        y_pred = model.predict_proba(set_x)
        if self.visualize:
            # Get the sample that gets the most positive prediction
            index_most_pos = np.argmax(y_pred, axis=0)[1]
            # Get the sample that gets the most negative prediction
            index_most_neg = np.argmin(y_pred, axis=0)[1]
            x_array_for_explain = set_x.values

            # Shap explainer
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(x_array_for_explain)
            # Get global explanation
            shap.summary_plot(shap_values, features=x_array_for_explain, feature_names=set_x.columns)
            # Get local explanation
            shap.force_plot(shap_explainer.expected_value, shap_values[index_most_neg],
                            features=x_array_for_explain[index_most_neg], feature_names=set_x.columns,
                            show=True, matplotlib=True)

            shap.force_plot(shap_explainer.expected_value, shap_values[index_most_pos],
                            features=x_array_for_explain[index_most_pos], feature_names=set_x.columns,
                            show=True, matplotlib=True)

            # Lime explainer
            lime_explainer = LimeTabularExplainer(x_array_for_explain,
                                                  feature_names=set_x.columns,
                                                  verbose=True,
                                                  mode='classification')
            # Get local explanation
            exp = lime_explainer.explain_instance(set_x.values[index_most_pos], model.predict_proba, num_features=10)
            exp.as_pyplot_figure()
            plt.tight_layout()
            plt.title((f'LIME for model: {self.model_name} on most positive test sample'))
            plt.show()
            plt.close()

            # Get local explanation
            exp = lime_explainer.explain_instance(set_x.values[index_most_neg], model.predict_proba, num_features=10)
            exp.as_pyplot_figure()
            plt.tight_layout()
            plt.title((f'LIME for model: {self.model_name} on most negative test sample'))
            plt.show()
            plt.close()

        metric_dict = get_all_metrics(y_gt=set_y['install'].values, y_pred=y_pred, visualize=self.visualize,
                                      model_name=self.model_name)
        return metric_dict, set_y['install'].values, y_pred

    def eval_on_valid(self):
        metric_dict, y_gt, y_pred = self.exec_eval(set_x=self.valid_set_x, set_y=self.valid_set_y)
        return metric_dict, y_gt, y_pred

    def eval_on_test(self):
        metric_dict, y_gt, y_pred = self.exec_eval(set_x=self.test_set_x, set_y=self.test_set_y)
        return metric_dict, y_gt, y_pred

    def generate_assessment_prediction(self):
        assessment_feature_df = pd.read_csv(self.preprocessing_settings.assessment_features_path, delimiter=';')[
            self.all_cols_and_id]
        model = pickle.load(open(self.best_model_output_path, "rb"))

        assessment_feature_df_x_id = assessment_feature_df[['id']].values
        assessment_feature_df_x = assessment_feature_df.drop(['id'], axis=1)
        y_pred_proba = model.predict_proba(assessment_feature_df_x)
        y_pred = np.expand_dims(y_pred_proba[:, 1], -1)
        result_values = np.concatenate([assessment_feature_df_x_id, y_pred], axis=-1)
        submission_df = pd.DataFrame(result_values, columns=['id',
                                                             'install_proba'])

        # generate count stats
        count_dict = return_counts_y_pred(y_pred=y_pred_proba)
        return submission_df, count_dict

    def clean_variables(self):
        self.training_df = None
        self.test_df = None
        self.valid_df = None
        self.training_set_y = None
        self.training_set_x = None
        self.valid_set_y = None
        self.valid_set_x = None
        self.test_set_y = None
        self.test_set_x = None
