import logging 
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning
from typing_extensions import Annotated

@step
def clean_data(data: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame , "x_train"],
    Annotated[pd.DataFrame , "x_test"],
    Annotated[pd.Series , "y_train"],
    Annotated[pd.Series , "y_test"]
]:
    try:
        data_cleaning = DataCleaning(data)
        df = data_cleaning.preprocess_data()
        x_train , x_test , y_train , y_test = data_cleaning.divide_data(df= df)
        return x_train , x_test , y_train , y_test

    except Exception as e:
        logging.error(e)
        raise e
    