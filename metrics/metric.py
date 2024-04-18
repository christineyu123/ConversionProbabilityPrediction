import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def return_auc(y_gt, y_pred, visualize, model_name):
    y_pred = y_pred[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_gt, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    if visualize:
        plt.plot(fpr, tpr, label='AUC = ' + str(round(roc_auc, 2)))
        plt.legend(loc='lower right')
        plt.title(f"ROC curve for model: {model_name}")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        print('auc showed')

    return roc_auc


def return_log_loss(y_gt, y_pred):
    log_loss = metrics.log_loss(y_true=y_gt, y_pred=y_pred)
    return log_loss


def return_accuracy(y_gt, y_pred):
    y_pred = y_pred[:, 1]
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = metrics.accuracy_score(y_true=y_gt, y_pred=y_pred)
    return accuracy


def return_precision_recall_f1(y_gt, y_pred):
    y_pred = y_pred[:, 1]
    y_pred = (y_pred > 0.5).astype(int)
    precision_micro, recall_micro, f1_micro, _ = metrics.precision_recall_fscore_support(y_true=y_gt, y_pred=y_pred,
                                                                                         average='micro')
    precision_binary, recall_binary, f1_binary, _ = metrics.precision_recall_fscore_support(y_true=y_gt, y_pred=y_pred,
                                                                                            average='binary')
    return precision_micro, recall_micro, f1_micro, precision_binary, recall_binary, f1_binary


def return_mcc(y_gt, y_pred):
    y_pred = y_pred[:, 1]
    y_pred = (y_pred > 0.5).astype(int)
    y_gt = y_gt.astype(int)
    res = metrics.matthews_corrcoef(y_true=y_gt, y_pred=y_pred)
    return res


def return_counts_y_gt(y_gt):
    y_gt = y_gt.astype(int)
    unique, counts = np.unique(y_gt, return_counts=True)
    count_dict = dict(zip(unique, counts))
    return {'0': int(count_dict[0]) if 0 in count_dict else 0,
            '1': int(count_dict[1]) if 1 in count_dict else 0,
            'total': len(y_gt)}


def return_counts_y_pred(y_pred):
    y_pred = y_pred[:, 1]
    y_pred = (y_pred > 0.5).astype(int)
    unique, counts = np.unique(y_pred, return_counts=True)
    count_dict = dict(zip(unique, counts))
    return {'0': int(count_dict[0]) if 0 in count_dict else 0,
            '1': int(count_dict[1]) if 1 in count_dict else 0,
            'total': len(y_pred)}


def confusion_ma(y_true, y_pred):
    y_pred = y_pred[:, 1]
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot(cmap=plt.cm.Blues)
    return plt.show()


def get_all_metrics(y_gt, y_pred, visualize, model_name):
    precision_micro, recall_micro, f1_micro, precision_binary, recall_binary, f1_binary = return_precision_recall_f1(
        y_gt, y_pred)
    result_dict = {
        'roc_auc': return_auc(y_gt, y_pred, visualize, model_name),
        'log_loss': return_log_loss(y_gt, y_pred),
        'accuracy': return_accuracy(y_gt, y_pred),
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_binary': f1_binary,
        'precision_binary': precision_binary,
        'recall_binary': recall_binary,
        'mcc': return_mcc(y_gt, y_pred),
        'y_gt_counts': return_counts_y_gt(y_gt=y_gt),
        'y_pred_counts': return_counts_y_pred(y_pred=y_pred)
    }
    return result_dict
