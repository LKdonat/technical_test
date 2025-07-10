import polars as pl 
import logging 

LOGGER = logging.getLogger(__name__)

def filter_unusable_columns(
    data: pl.DataFrame, 
    columns_to_remove: list
) -> pl.DataFrame:
    """This function removes columns that are deemed unusable (ex : GDPR, unexploitable information,...)

    Args:
        data (pl.DataFrame): _description_
        columns_to_remove (list): _description_

    Returns:
        pl.DataFrame: _description_
    """
    LOGGER.info(f"{data.shape = } : {data['policy_number'].n_unique() = }")
    filtered_data = (
        data
        .select(
            [col for col in data.columns if col not in columns_to_remove]
        )
    )
    LOGGER.info(f"{filtered_data.shape = } : {filtered_data['policy_number'].n_unique() = }")
    return filtered_data

def convert_dates_str_to_datetime(data: pl.DataFrame, date_columns_list: list) -> pl.DataFrame :
    """This function converts the date as str columns to datetime
    This will allow us to calculate timedeltas

    Args:
        data (pl.DataFrame): _description_
        date_columns_list (list): _description_

    Returns:
        pl.DataFrame: _description_
    """
    data_with_datetime_columns = data.with_columns([
        pl.col(col).str.strptime(pl.Date, "%Y-%m-%d") for col in date_columns_list
    ])
    return data_with_datetime_columns

def split_csl_column(data: pl.DataFrame) -> pl.DataFrame:
    """This function splits the 'policy_csl' column into 2 columns
    Business logic : 
        - CSL means Combined Single Limit but the column contains a ratio
        - it is very likely that it refers to a split limits (per person / per accident)
        - Example with "250/500":
            250 = $250,000 maximum per injured person.
            500 = $500,000 maximum per accident for all injured people combined.

    Args:
        data (pl.DataFrame): _description_

    Returns:
        pl.DataFrame: _description_
    """
    data = data.with_columns([
        pl.col("policy_csl").str.split("/").list.get(0).cast(pl.Float64).alias("per_person_limit"),
        pl.col("policy_csl").str.split("/").list.get(1).cast(pl.Float64).alias("per_accident_limit"),
    ])
    LOGGER.info(f"{data.shape = } : {data['policy_number'].n_unique() = }")
    return data.drop("policy_csl")

def calculate_timedelta_between_incident_date_and_bind_date(data:pl.DataFrame) -> pl.DataFrame :
    """This function calculates the number of years between the incident date and the bind date
    Business logic : customers that subscribe to a policy not long before declaring a claim are usually suspicious

    Args:
        data (pl.DataFrame): _description_

    Returns:
        pl.DataFrame: _description_
    """
    data = (
        data
        .with_columns(
            (
                (pl.col("incident_date") - pl.col("policy_bind_date")).cast(pl.Float64) / 365.25
            )
            .alias("nb_years_between_incident_and_bind_date")
        )
    )
    LOGGER.info(f"{data.shape = } : {data['policy_number'].n_unique() = }")
    return data

def split_instances_labels(data: pl.DataFrame, target_col: str, id_col: str)-> tuple:
    LOGGER.info(f"{data.shape = }")
    labels = data.select([id_col, target_col])
    LOGGER.info(f"{labels.shape = }")
    instances = data.drop([target_col])
    LOGGER.info(f"{instances.shape = }")
    return instances, labels