import mlflow
import xgboost as xgb 

class CustomXGBoostMLFlowLoggerCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model: xgb.Booster, epoch:int, evals_log: dict) -> bool:
        for (
            data_name,
            metric_dict,
        ) in evals_log.items() :
            for metric_name, metric_values in metric_dict.items():
                metric_value = metric_values[-1]
                mlflow.log_metric(
                    key = f"{data_name}_{metric_name}", 
                    value = metric_value, 
                    step=epoch
                )