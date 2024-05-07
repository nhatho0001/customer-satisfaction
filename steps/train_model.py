import logging
import pandas as pd
import mlflow
from zenml import step
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from src.model_dev import ModelTrainer
from zenml.client import Client 
from typing import Annotated 
from zenml import ArtifactConfig

experiment_tracker = Client().active_stack.experiment_tracker
@step(enable_cache=False,experiment_tracker= experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame ,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str = "lightgbm",
    do_fine_tuning: bool = True
)-> Annotated[
    RegressorMixin,
    ArtifactConfig(name="sklearn_regressor", is_model_artifact=True),
]:
    model_training = ModelTrainer(x_train= x_train , y_train= y_train , x_test=x_test , y_test=y_test)
    try:
        if model_type == "lightgbm":
            mlflow.lightgbm.autolog()
            lightGBM = model_training.lightgbm_trainer(do_fine_tuning)
            return lightGBM
        elif model_type == "randomforest":
            mlflow.sklearn.autolog()
            rf_model = model_training.random_forest_trainer(
                fine_tuning=do_fine_tuning
            )
            return rf_model
        elif model_type == "xgboost":
            mlflow.xgboost.autolog()
            xgb_model = model_training.xgboost_trainer(
                fine_tuning=do_fine_tuning
            )
            return xgb_model
        else:
            raise ValueError("Model type not supported")
    except Exception as e:
        logging.error(e)
        raise e
        

