import pandas as pd
import logging 

from src.data_cleaning import DataCleaning 

def get_data_for_test(): 
    try:
        df = pd.read_csv("D:\Project\data\olist_customers_dataset.csv")
        df = df.sample(100)
        data_clean = DataCleaning(df)
        df = data_clean.preprocess_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e