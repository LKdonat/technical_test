import pandas as pd 
import logging 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils.custom_xgb_mlflow_logger_callback import CustomXGBoostMLFlowLoggerCallback

LOGGER = logging.getLogger(__name__)

def apply_train_test_split(x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(x, y, **kwargs)
    return x_train, x_test, y_train, y_test

def train_xgb_model(
    x_train: pd.DataFrame, 
    x_test: pd.DataFrame, 
    y_train: pd.DataFrame, 
    y_test: pd.DataFrame, 
    hyperparameters: dict, 
    training_params : dict
) -> xgb.XGBClassifier:
    early_stopping_callback = xgb.callback.EarlyStopping(
        rounds = training_params["early_stopping_rounds"], save_best=True
    )
    mlflow_logger_callback = CustomXGBoostMLFlowLoggerCallback()
    hyperparameters.update(dict(callbacks=[early_stopping_callback, mlflow_logger_callback]))
    xgb_classifier = xgb.XGBClassifier(**hyperparameters).fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_test, y_test)]
    )
    return xgb_classifier
