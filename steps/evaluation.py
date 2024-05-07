import logging
import pandas as pd
import mlflow
from zenml import step
from typing import Tuple 
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from src.evaluate_model import MSE , R2Score, RMSE
from zenml.client import Client 

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker= experiment_tracker.name)
def Evaluation(model: RegressorMixin,
               x_test: pd.DataFrame,
               y_test: pd.DataFrame,)-> Tuple[
                   Annotated[float , "r2_score"],
                   Annotated[float , "rmse"]
               ]:
    try:
        predict = model.predict(x_test)
        r2_score = R2Score().calculate_score(y_true=y_test , y_pred= predict)
        mlflow.log_metric("r2_score" ,r2_score)
        mse = MSE().calculate_score(y_true=y_test , y_pred= predict)
        mlflow.log_metric("mse" , mse)
        rmse = RMSE().calculate_score(y_true= y_test , y_pred= predict)
        mlflow.log_metric("rsme",rmse)
        return r2_score , rmse


    except Exception as e:
        logging.error(e)
        raise e