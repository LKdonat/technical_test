import pandas as pd 
import logging 
import numpy as np 
from sklearn.preprocessing import OrdinalEncoder
from typing import Union

LOGGER = logging.getLogger(__name__)

def set_index(data: pd.DataFrame) -> pd.DataFrame: 
    data.set_index("policy_number", inplace=True)
    return data

def remove_unused_columns(data: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """This function removes the columns that aren't needed for modeling such as date columns

    Args:
        data (pd.DataFrame): _description_
        columns_to_remove (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    filtered_data = data[list(set(data.columns) - set(columns_to_remove))].copy()
    return filtered_data

def set_features_type(data: pd.DataFrame, features_type_mapping: dict) -> pd.DataFrame:
    typed_data = data.astype(features_type_mapping)
    return typed_data 

def impute_missing_values(
    data: pd.DataFrame, 
    binary_columns_list: list, 
    imputation_rules: dict = None, 
) -> pd.DataFrame:
    """This function replaces the missing values ("?") observed in the categorical data, we have 2 cases : 
        - Binary columns (Yes/No) -> here we will replace "?" by NO because for binary columns most of the time missing value means NO
        - Categorical columns with more than 2 categories -> here we will leave "?" as it will create another category for unknown values
    There are no observed NAs in the dataset but we will leave an option to fill the potential NAs during inference with the imputation_rules dict

    Args:
        data (pd.DataFrame): _description_
        imputation_rules (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data = data.fillna(imputation_rules) if imputation_rules is not None else data.copy()
    for col in binary_columns_list:
        data.loc[:, col].replace(to_replace="?", value="NO")
    return data

def fit_columns_order(data: pd.DataFrame) -> pd.Index:
    """Here we fit the columns order because during inference the model will expect to have the exact columns in the exact order than during training
    So during training we fit and transform the columns order
    During inference we only transform the columns order with what we have fitted during training

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.Index: _description_
    """
    columns_order = data.columns
    return columns_order 

def transform_columns_order(data: pd.DataFrame, columns_order: pd.DataFrame) -> pd.DataFrame:
    data = data[columns_order]
    return data

def split_numeric_and_categorical_data(data: pd.DataFrame) -> tuple:
    """Here we split the numerical and categorical data because they will have separated processing steps before re aggregating them later on
    Especially categorical data that will be encoded

    Args:
        data (pd.DataFrame): _description_

    Returns:
        tuple: _description_
    """
    numerical_data = data.select_dtypes(include=[np.number])
    LOGGER.info(f"{numerical_data.columns = }")
    categorical_data = data.select_dtypes(include=['category', 'object'])
    LOGGER.info(f"{categorical_data.columns = }")
    return numerical_data, categorical_data 

def _apply_new_category(
    categorical_value: str, 
    categorical_mapping: dict
) -> str: 
    """This function takes a mapping following this format : 
    {
        "value_1": "group_1"
        "value_2": "group_1"
        ...
        "value_n": "group_3"
    }

    Then uses this mapping to group categorical columns with high cardinality following a logical grouping

    Args:
        categorical_value (str): _description_
        categorical_mapping (dict): _description_

    Returns:
        str: _description_
    """
    formated_category = categorical_mapping.get(categorical_value, "OTHER")
    return formated_category

def group_high_cardinality_categorical_columns_values(data: pd.DataFrame, all_categorical_mappings: dict) -> pd.DataFrame: 
    """This function takes all_categories_mappings as a parameter, it has the following format :
    {
        "col_1" : {
            "value_1": "group_1"
            "value_2": "group_1"
            ...
            "value_n": "group_3"
        },
        ...
        "col_n" : {
            "value_1": "group_1"
            "value_2": "group_1"
            ...
            "value_n": "group_3"
        },
    }
    Then uses it to group categorical columns with high cardinality into logical groups before encoding

    Args:
        data (pd.DataFrame): _description_
        all_categorical_mappings (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    for col, col_mapping in all_categorical_mappings.items():
        LOGGER.info(f"Grouping column '{col}' categories with the following mapping : {col_mapping}")
        data.loc[:, col] = (
            data[col]
            .apply(
                lambda x : _apply_new_category(
                    categorical_value= str(x), 
                    categorical_mapping=col_mapping
                )
            )
        )
        LOGGER.info(f"New values for column '{col}' : {data[col].value_counts(dropna=False).to_dict()}")
    return data

def fit_categorical_encoder(data: pd.DataFrame, unknown_value: Union[int, float]) -> OrdinalEncoder:
    """In this function we fit the categorical encoder.
    Why don't wee use fit_transform ? 
    -> during inference we wan't the exact encoder that was fitted during training so that there is no risk of the categories changing encoded value if new categories appear in the future
    -> during training we fit then transform 
    -> during inference we transform using the fitted encoder

    Why do we use OrdinalEncoder and not Target or OneHot ?
        - Due to XGBoost tree-based nature it won't have any negative effect to choose ordinal encoder because trees split the data using thresholds
        - OneHot will increase dimensionnality tremendously 
        - Target can lead to data leakage

    Args:
        data (pd.DataFrame): _description_
        unknown_value (Union[int, float]): _description_

    Returns:
        OrdinalEncoder: _description_
    """
    encoder =OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unknown_value)
    encoder.fit(data)
    return encoder

def transform_categorical_encoder(data: pd.DataFrame, encoder: OrdinalEncoder) -> pd.DataFrame :
    encoded_data = pd.DataFrame(
        data = encoder.transform(data),
        columns = data.columns,
        index = data.index
    )
    return encoded_data

def concatenate_numerical_and_categorical_data(
    numerical_data: pd.DataFrame, 
    categorical_data: pd.DataFrame
) -> pd.DataFrame:
    return pd.concat([numerical_data, categorical_data], axis = 1)

def create_response_variable(labels: pd.DataFrame) -> pd.DataFrame:
    labels.loc[labels.fraud_reported.str.upper().isin(["Y", "Yes"]), "tp_fraud"] = 1
    labels.loc[:, "tp_fraud"] = labels.tp_fraud.fillna(0)
    LOGGER.info(f"{labels.tp_fraud.value_counts().to_dict() = }")
    return labels[["tp_fraud"]]