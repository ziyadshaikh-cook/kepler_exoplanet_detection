import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def run_pipeline(self):
        try:
            logging.info("========== Training Pipeline Started ==========")

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test, _ = transformation.initiate_data_transformation(
                train_path, test_path
            )

            # Step 3: Model Training
            trainer = ModelTrainer()
            model_path = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

            logging.info("========== Training Pipeline Completed ==========")
            return model_path

        except Exception as e:
            raise CustomException(e, sys)