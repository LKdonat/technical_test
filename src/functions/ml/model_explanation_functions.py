import pandas as pd 
import logging 
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import shap 

LOGGER = logging.getLogger(__name__)

def create_shap_values(data: pd.DataFrame, model: xgb.XGBClassifier) -> pd.DataFrame:
    shap_values = pd.DataFrame(
        data = model.get_booster().predict(data=xgb.DMatrix(data), pred_contribs=True)[:,:-1],
        columns = data.columns,
        index=data.index
    )
    return shap_values

def plot_importance_variables(model: xgb.XGBClassifier, data: pd.DataFrame) :
    importances_df = pd.DataFrame(
        {
            "features": data.columns,
            "importance" : model.feature_importances_
        }
    ).sort_values(by="importance", ascending = True)
    plt.figure(figsize=(10,12))
    bars = plt.barh(importances_df["features"], importances_df["importance"], color="red")
    for bar, importance in zip(bars, importances_df["importance"]):
        plt.text(
            bar.get_width() + 0.01, 
            bar.get_y() + bar.get_height() / 2, 
            f"{importance:.8f}", 
            va="center", 
            fontsize=9
        )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature importance")
    img = plt.gcf()
    plt.close()
    return img

def plot_dependence_plots(
        data: pd.DataFrame,
        shap_values: pd.DataFrame,
        encoder: OrdinalEncoder
) -> dict: 
    data.loc[:, encoder.feature_names_in_] = encoder.inverse_transform(data[encoder.feature_names_in_])
    shap_values = shap_values.loc[:,data.columns]
    sns.set_theme()
    dict_img = dict()
    columns_by_importance = data.columns[np.argsort(-np.abs(shap_values).mean(0))]
    for i in range(data.shape[1]):
        shap.dependence_plot(
            ind="rank(" +str(i)+")",
            ax = plt.figure(figsize = (5,5)).gca(),
            shap_values=shap_values.values,
            features=data,
            interaction_index=None,
            show=False,
            alpha= 0.1,
            color="red",
            feature_names=data.columns,
        )
        plt.axhline(0,color="k")
        plt.xticks(rotation=45)
        img = plt.gcf()
        dict_img[f"{i:02d}_{columns_by_importance[i]}.png"] = img 
        plt.close()
    return dict_img
