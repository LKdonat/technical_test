import pandas as pd 
import logging 
import xgboost as xgb

LOGGER = logging.getLogger(__name__)

def predict_probs(data:pd.DataFrame, model: xgb.XGBClassifier)->pd.DataFrame:
    y_probs = pd.DataFrame(
        data=model.predict_proba(data, validate_features=True)[:,1],
        index=data.index,
        columns=["proba"]
    )
    return y_probs