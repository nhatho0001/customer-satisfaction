import json

# from .utils import get_data_for_test
import os
import logging
import numpy as np
import pandas as pd
#from materializer.custom_materializer import cs_materializer
from pipelines.until import get_data_for_test
from steps.ingest_data import load_data 
from steps.clean_data import clean_data 
from steps.train_model import train_model
from steps.evaluation import Evaluation
from steps.model_loader import model_loader
from steps.predicter import predictor
from steps.prediction_service_loader import prediction_service_loader
from pipelines.train_pipelines import train_pipeline
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

docker_setting = DockerSettings(required_integrations=[MLFLOW])
class DeploymentTrigerConfig(BaseParameters):
    min_accuracy = 0.92 

@step 
def deployment_triger(
    accuracy: float ,
    config: DeploymentTrigerConfig
) -> bool:
    return accuracy >= config.min_accuracy
@pipeline
def continuous_deployment_pipeline(path:str,
                                 model_type: str = "lightgbm" , 
                                 )-> None:
    train_pipeline(path = path , model_type = model_type)
    model = model_loader(model_name = "Customer_Satisfaction_Predictor" ,
                         after= 'model_promoter')
    mlflow_model_deployer_step(model = model,
                               workers=3, 
                               deploy_decision=True)


@pipeline(enable_cache=False)
def inference_pipeline()-> None:
    bach_data = get_data_for_test()
    predict_model = prediction_service_loader(pipeline_name = "continuous_deployment_pipeline" ,
                                              step_name = "mlflow_model_deployer_step",
                                              running = False)
    
    predictor(service=predict_model , input_data = bach_data )
    
