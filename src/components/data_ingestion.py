import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # Only filter — remove CANDIDATE rows, keep CONFIRMED and FALSE POSITIVE
            df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
            logging.info(f"After filtering CANDIDATE rows: {df.shape[0]} rows remaining")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train size: {train_set.shape[0]} | Test size: {test_set.shape[0]}")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test files saved to artifacts/")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Train:", train_path)
    print("Test:", test_path)