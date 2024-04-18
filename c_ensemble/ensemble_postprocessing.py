import numpy as np

from metrics.metric import get_all_metrics


def generate_ensemble_for_eval(models, model_name, y_gts, y_preds):
    ensemble_y_gts_valid = y_gts[
        models[0]]  # y_gts should be same across all model, so just take the first one saved
    ensemble_y_preds_valid = list(y_preds.values())
    ensemble_y_preds_valid = np.array([np.expand_dims(data[:, 1], -1) for data in ensemble_y_preds_valid])
    ensemble_y_preds_valid = np.concatenate(ensemble_y_preds_valid, axis=-1).mean(axis=1)
    ensemble_y_preds_valid_0 = 1 - ensemble_y_preds_valid
    ensemble_y_preds_valid = np.stack([ensemble_y_preds_valid_0, ensemble_y_preds_valid], axis=1)
    metric_dict = get_all_metrics(y_gt=ensemble_y_gts_valid, y_pred=ensemble_y_preds_valid, visualize=False,
                                  model_name=model_name)
    return ensemble_y_gts_valid, ensemble_y_preds_valid, metric_dict
