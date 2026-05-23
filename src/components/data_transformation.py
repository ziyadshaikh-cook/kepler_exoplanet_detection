import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, feature_columns):
        """
        Builds and returns the preprocessing pipeline.
        Fitted on train data only — then applied to both train and test.
        """
        try:
            skewed_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_insol']
            # Only include skewed cols that are actually present in features
            skewed_cols = [col for col in skewed_cols if col in feature_columns]
            normal_cols = [col for col in feature_columns if col not in skewed_cols]

            skewed_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("log_transform", FunctionTransformer(np.log1p, validate=True)),
                ("scaler", RobustScaler())
            ])

            normal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("skewed_cols", skewed_pipeline, skewed_cols),
                ("normal_cols", normal_pipeline, normal_cols)
            ])

            logging.info(f"Skewed columns (log+scale): {skewed_cols}")
            logging.info(f"Normal columns (scale only): {normal_cols}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            # --- Drop metadata and irrelevant columns ---
            cols_to_drop = [
                'rowid', 'kepid', 'kepoi_name', 'kepler_name',
                'koi_pdisposition', 'ra', 'dec', 'koi_tce_delivname',
                'koi_teq_err1', 'koi_teq_err2',
                'koi_score'
            ]
            train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
            test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
            logging.info(f"After dropping irrelevant columns — Train: {train_df.shape} | Test: {test_df.shape}")

            # --- Encode target ---
            target_column = "koi_disposition"
            train_df[target_column] = (train_df[target_column] == "CONFIRMED").astype(int)
            test_df[target_column] = (test_df[target_column] == "CONFIRMED").astype(int)
            logging.info("Target encoded: CONFIRMED=1, FALSE POSITIVE=0")

            # --- Separate features and target ---
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # --- Build and fit preprocessor on train only ---
            preprocessor = self.get_data_transformer_object(X_train.columns.tolist())
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Preprocessing complete — fit on train, transform applied to both")

            # --- Save preprocessor ---
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info(f"Preprocessor saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train.values,
                y_test.values,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:", pd.Series(y_train).value_counts().to_dict())
    print("Preprocessor saved at:", preprocessor_path)