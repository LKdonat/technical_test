import pandas as pd 
import logging 
import seaborn as sns
import sklearn.metrics as metrics
import numpy as np

LOGGER = logging.getLogger(__name__)

def _create_classification_report(
        y_true:pd.DataFrame, y_preds: pd.DataFrame
) -> dict: 
    """This function evaluates the model, the metrics chosen for evaluation are :
    - precision
    - recall
    - the confusion matrix 
    we have chosen these metrics because for fraud detection what matters the most is the nb of fraud detected and the amount of cases the model predicts as a high risk of fraud

    Args:
        y_true (pd.DataFrame): _description_
        y_preds (pd.DataFrame): _description_

    Returns:
        dict: _description_
    """
    classification_report = dict(
        precision = metrics.precision_score(y_true = y_true.tp_fraud, y_pred=y_preds.preds, zero_division=0), 
        recal = metrics.recall_score(y_true = y_true.tp_fraud, y_pred=y_preds.preds, zero_division=0), 
        true_positives = float(np.sum((y_preds.preds == 1) & (y_true.tp_fraud == 1))),
        false_positives = float(np.sum((y_preds.preds == 1) & (y_true.tp_fraud == 0))),
        true_negatives = float(np.sum((y_preds.preds == 0) & (y_true.tp_fraud == 0))),
        false_negatives = float(np.sum((y_preds.preds == 0) & (y_true.tp_fraud == 1))),
    )
    return classification_report

def create_classification_reports(y_true:pd.DataFrame, y_probs: pd.DataFrame, thresholds: list) -> dict:
    all_metrics = {}
    for threshold in thresholds :
        y_preds = ((y_probs >= threshold) * 1).rename(columns = {"proba": "preds"})
        all_metrics[f"classification_report_threshold_{int(100*threshold)}%"] = _create_classification_report(y_true = y_true, y_preds=y_preds)
    return all_metrics