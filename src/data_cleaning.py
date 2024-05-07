import logging 
import pandas as pd
import numpy as np 
from typing import Union 
from abc import ABC , abstractmethod 
from sklearn.model_selection import train_test_split

class DataCleaning:
    def __init__(self , data: pd.DataFrame)-> None:
        self.df = data
    
    def preprocess_data(self) -> pd.DataFrame:
        try:
            self.df = self.df.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            self.df["product_weight_g"].fillna(
                self.df["product_weight_g"].median(), inplace=True
            )
            self.df["product_length_cm"].fillna(
                self.df["product_length_cm"].median(), inplace=True
            )
            self.df["product_height_cm"].fillna(
                self.df["product_height_cm"].median(), inplace=True
            )
            self.df["product_width_cm"].fillna(
                self.df["product_width_cm"].median(), inplace=True
            )
            self.df["review_comment_message"].fillna("No review", inplace=True)

            self.df = self.df.select_dtypes(include=[np.number])
            cols_to_drop = [
                "customer_zip_code_prefix",
                "order_item_id",
            ]
            self.df = self.df.drop(cols_to_drop, axis=1)

            # Catchall fillna in case any where missed
            self.df.fillna(self.df.mean(), inplace=True)

            return self.df
        except Exception as e:
            logging.error(e)
            raise e
    
    def divide_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            X = df.drop("review_score", axis=1)
            y = df["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e