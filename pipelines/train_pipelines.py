from zenml import pipeline
from steps.ingest_data import load_data 
from steps.clean_data import clean_data 
from steps.train_model import train_model
from steps.evaluation import Evaluation
from steps.model_promoter import model_promoter
from steps.config import ModelNameConfig
from zenml.client import Client 


@pipeline
def train_pipeline(path: str , model_type: str = "lightgbm"):
    df = load_data(path_file = path)
    X_train , X_test , Y_train , Y_test = clean_data(df)
    model = train_model(X_train , X_test , Y_train , Y_test , model_type = model_type)
    r2_score , mse = Evaluation(model , X_test , Y_test)
    is_promoted = model_promoter(mse = mse)
    return model , is_promoted
    


    
