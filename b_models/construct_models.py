from enum import Enum

from a_preprocessing.explore_preprocessor import ExplorePreprocessorSettings
from a_preprocessing.naive_preprocessor import NaivePreprocessorSettings
from b_models.dlrm_model import DLRMModel
from b_models.xgboost_model import XGBoostModel


class ModelChoices(Enum):
    # Use data from ExplorePreprocessor
    dlrm_naive_explore = 'dlrm_naive_explore'
    dlrm_resample_explore = 'dlrm_resample_explore'
    dlrm_class_weight_explore = 'dlrm_class_weight_explore'
    dlrm_focal_loss_explore = 'dlrm_focal_loss_explore'
    dlrm_label_smoothing_explore = 'dlrm_label_smoothing_explore'
    dlrm_feature_importance_explore = 'dlrm_feature_importance_explore'
    xgboost_naive_explore = 'xgboost_naive_explore'
    xgboost_resample_explore = 'xgboost_resample_explore'
    # Use data from NaivePreprocessor
    dlrm_naive = 'dlrm_naive'
    dlrm_resample = 'dlrm_resample'
    dlrm_class_weight = 'dlrm_class_weight'
    dlrm_focal_loss = 'dlrm_focal_loss'
    dlrm_label_smoothing = 'dlrm_label_smoothing'
    dlrm_feature_importance = 'dlrm_feature_importance'
    xgboost_naive = 'xgboost_naive'
    xgboost_resample = 'xgboost_resample'
    # Combo with many flags enabled
    dlrm_combo_explore = 'dlrm_combo_explore'


def get_models(models_names, raw_train_data_path):
    all_models = []
    for name in models_names:
        if str(name).endswith('explore'):
            preprocessing_settings = ExplorePreprocessorSettings()
        else:
            preprocessing_settings = NaivePreprocessorSettings()

        if name == ModelChoices.dlrm_naive.value or name == ModelChoices.dlrm_naive_explore.value:
            cur_model = DLRMModel(preprocessing_settings=preprocessing_settings,
                                  model_name=name,
                                  raw_train_data_path=raw_train_data_path)
        elif name == ModelChoices.dlrm_resample.value or name == ModelChoices.dlrm_resample_explore.value:
            cur_model = DLRMModel(preprocessing_settings=preprocessing_settings,
                                  model_name=name,
                                  is_resample=True,
                                  raw_train_data_path=raw_train_data_path)
        elif name == ModelChoices.dlrm_class_weight.value or name == ModelChoices.dlrm_class_weight_explore.value:
            cur_model = DLRMModel(preprocessing_settings=preprocessing_settings,
                                  model_name=name,
                                  is_class_weights=True,
                                  raw_train_data_path=raw_train_data_path)
        elif name == ModelChoices.dlrm_focal_loss.value or name == ModelChoices.dlrm_focal_loss_explore.value:
            cur_model = DLRMModel(preprocessing_settings=preprocessing_settings,
                                  model_name=name,
                                  is_focal_loss=True,
                                  raw_train_data_path=raw_train_data_path)
        elif name == ModelChoices.dlrm_label_smoothing.value or name == ModelChoices.dlrm_label_smoothing_explore.value:
            cur_model = DLRMModel(preprocessing_settings=preprocessing_settings,
                                  model_name=name,
                                  is_label_smoothing=True,
                                  raw_train_data_path=raw_train_data_path)
        elif name == ModelChoices.dlrm_feature_importance.value or name == ModelChoices.dlrm_feature_importance_explore.value:
            cur_model = DLRMModel(preprocessing_settings=preprocessing_settings,
                                  model_name=name,
                                  is_feature_importance=True,
                                  feature_importance_path='./XGBoostModel_output_folder/xgboost_naive_explore_top_N_features.json',
                                  raw_train_data_path=raw_train_data_path)
        elif name == ModelChoices.xgboost_naive.value or name == ModelChoices.xgboost_naive_explore.value:
            cur_model = XGBoostModel(preprocessing_settings=preprocessing_settings,
                                     model_name=name, is_resample=False)
        elif name == ModelChoices.xgboost_resample.value or name == ModelChoices.xgboost_resample_explore.value:
            cur_model = XGBoostModel(preprocessing_settings=preprocessing_settings,
                                     model_name=name, is_resample=True)
        elif name == ModelChoices.dlrm_combo_explore.value:
            cur_model = DLRMModel(preprocessing_settings=preprocessing_settings,
                                  model_name=name,
                                  is_focal_loss=False,
                                  is_resample=False,
                                  is_class_weights=True,
                                  is_label_smoothing=True,
                                  is_feature_importance=True,
                                  raw_train_data_path=raw_train_data_path)
        else:
            raise ValueError("Unknown model")
        all_models.append(cur_model)

    return all_models
