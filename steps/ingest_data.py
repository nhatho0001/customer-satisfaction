import logging
import pandas as pd
from zenml import step


class Ingest_data:
    def __init__(self , path_file) -> None:
        self.path_file = path_file
    
    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from {self.path_file}")
        return pd.read_csv(self.path_file)
    

@step
def load_data(path_file: str)->pd.DataFrame:
    
    try:
        ingest_data = Ingest_data(path_file)
        data = ingest_data.get_data()
        return data
    except Exception as e:
        logging.error(f"Error while ingesting data {e}")
        raise e

