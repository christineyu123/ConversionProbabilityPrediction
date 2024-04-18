import json
import os
import time

import tensorflow as tf

from a_preprocessing.construct_preprocessors import get_preprocessors
from b_models.construct_models import get_models, ModelChoices
from c_ensemble.ensemble_postprocessing import generate_ensemble_for_eval

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

temp_final_result_folder = 'final_result_runtime_{}'.format(int(round(time.time() * 1000)))

if not os.path.exists(temp_final_result_folder):
    os.makedirs(temp_final_result_folder)

train_data_path = './data/train_data.csv'
assessment_data_path = './data/assessment_data.csv'

IS_PREPROCESS = True
IS_TRAIN = True

if IS_PREPROCESS:
    # get preprocessors
    preprocessors = get_preprocessors(preprocessors_names=['naive', 'explore'],
                                      train_data_path=train_data_path,
                                      assessment_data_path=assessment_data_path)
    # run preprocessors
    for cur_preprocessor in preprocessors:
        cur_preprocessor.generate_features()

# get models
models_names = [
    ModelChoices.xgboost_naive_explore.value,
    ModelChoices.xgboost_resample_explore.value,
    ModelChoices.dlrm_naive_explore.value,
    ModelChoices.dlrm_resample_explore.value,
    ModelChoices.dlrm_class_weight_explore.value,
    ModelChoices.dlrm_focal_loss_explore.value,
    ModelChoices.dlrm_label_smoothing_explore.value,
    ModelChoices.dlrm_feature_importance_explore.value,

    ModelChoices.xgboost_naive.value,
    ModelChoices.xgboost_resample.value,
    ModelChoices.dlrm_naive.value,
    ModelChoices.dlrm_resample.value,
    ModelChoices.dlrm_class_weight.value,
    ModelChoices.dlrm_focal_loss.value,
    ModelChoices.dlrm_label_smoothing.value,
    ModelChoices.dlrm_feature_importance.value,

    ModelChoices.dlrm_combo_explore.value
]

# define ensemble names
ensemble_models_names = [
    ModelChoices.dlrm_naive_explore.value,
    ModelChoices.dlrm_resample_explore.value,
    ModelChoices.dlrm_class_weight_explore.value,
]

models = get_models(models_names=models_names,
                    raw_train_data_path=train_data_path)

# run models
model_results_valid = {}
y_gts_valid = {}
y_preds_valid = {}
model_results_test = {}
y_gts_test = {}
y_preds_test = {}
for cur_model in models:
    cur_model.split_data()
    if IS_TRAIN:
        cur_model.train()
    cur_valid_metric_dict, cur_y_gt_valid, cur_y_pred_valid = cur_model.eval_on_valid()

    model_results_valid[cur_model.model_name] = cur_valid_metric_dict
    if cur_model.model_name in ensemble_models_names:
        y_gts_valid[cur_model.model_name] = cur_y_gt_valid
        y_preds_valid[cur_model.model_name] = cur_y_pred_valid

    cur_test_metric_dict, cur_y_gt_test, cur_y_pred_test = cur_model.eval_on_test()

    model_results_test[cur_model.model_name] = cur_test_metric_dict
    if cur_model.model_name in ensemble_models_names:
        y_gts_test[cur_model.model_name] = cur_y_gt_test
        y_preds_test[cur_model.model_name] = cur_y_pred_test
    # clean model
    cur_model.clean_variables()
    # dump intermediate model results in case subsequent models break
    json.dump(model_results_valid, open(os.path.join(temp_final_result_folder, "model_results_valid.json"), "w"))
    json.dump(model_results_test, open(os.path.join(temp_final_result_folder, "model_results_test.json"), "w"))

# generate an ensemble of ALL models using average prediction
ensemble_model_name = "ensemble"

ensemble_y_gts_valid, ensemble_y_preds_valid, ensemble_metric_dict_valid = generate_ensemble_for_eval(
    models=ensemble_models_names,
    model_name=ensemble_model_name,
    y_gts=y_gts_valid,
    y_preds=y_preds_valid)
model_results_valid[ensemble_model_name] = ensemble_metric_dict_valid
y_gts_valid[ensemble_model_name] = ensemble_y_gts_valid
y_preds_valid[ensemble_model_name] = ensemble_y_preds_valid

ensemble_y_gts_test, ensemble_y_preds_test, ensemble_metric_dict_test = generate_ensemble_for_eval(
    models=ensemble_models_names,
    model_name=ensemble_model_name,
    y_gts=y_gts_test,
    y_preds=y_preds_test)
model_results_test[ensemble_model_name] = ensemble_metric_dict_test
y_gts_test[ensemble_model_name] = ensemble_y_gts_test
y_preds_test[ensemble_model_name] = ensemble_y_preds_test

print(f"model_results_valid: {json.dumps(model_results_valid, indent=2)}")
print(f"model_results_test: {json.dumps(model_results_test, indent=2)}")
json.dump(model_results_valid, open(os.path.join(temp_final_result_folder, "model_results_valid.json"), "w"))
json.dump(model_results_test, open(os.path.join(temp_final_result_folder, "model_results_test.json"), "w"))

# generate submission
submission_df_records = []
for final_model in models:
    submission_df, count_stats_dict = final_model.generate_assessment_prediction()
    submission_df.to_csv(os.path.join(temp_final_result_folder, f"{final_model.model_name}_submission.csv"), sep=';',
                         index=False)

    print(f"Submission count_stats_dict from model {final_model.model_name} is: {count_stats_dict}")
    json.dump(count_stats_dict, open(
        os.path.join(temp_final_result_folder, f"submission_count_stats_for_model_{final_model.model_name}.json"), "w"))
    if final_model.model_name in ensemble_models_names:
        submission_df_records.append(submission_df)
    final_model.clean_variables()

# generate submission using ensemble
i = 0
preds = submission_df_records[i].rename(columns={'install_proba': f'install_proba_{i}'})
for i, d in enumerate(submission_df_records[1:]):
    preds = preds.merge(d, how='left', on='id')

preds['install_proba'] = preds[[c_name for c_name in preds.columns if 'install_proba' in c_name]].mean(axis=1)
preds = preds[['id', 'install_proba']]
preds.to_csv(os.path.join(temp_final_result_folder, f"ensemble_model_submission.csv"), sep=';',
             index=False)
print("All done!")
