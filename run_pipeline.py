from pipelines.train_pipelines import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    train_pipeline.with_options(
            config_path="config.yaml"
        )(path = "D:\Project\data\olist_customers_dataset.csv" , model_type = "xgboost")
    uri_tracker = get_tracking_uri()
    print(f"Uri of Mlflow Ui : {uri_tracker}")
    "bash :  mlflow ui --backend-store-uri uri_tracker"