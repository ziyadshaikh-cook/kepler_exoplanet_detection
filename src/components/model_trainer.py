import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import dagshub
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    classification_report
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model training started")

            # --- DagsHub + MLflow setup ---
            dagshub.init(
                repo_owner="ziyadshaikh-cook",
                repo_name="kepler_exoplanet_detection",
                mlflow=True
            )

            models = {
                "XGBoost": XGBClassifier(
                    scale_pos_weight=4069/1783,
                    random_state=42,
                    eval_metric='logloss'
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )
            }

            params = {
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.1, 0.2],
                    "subsample": [0.8, 1.0]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.2]
                },
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0]
                }
            }

            # --- Train and evaluate all models ---
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # --- Pick best model by F1 score ---
            best_model_name = max(model_report, key=lambda k: model_report[k]["f1_score"])
            best_model_score = model_report[best_model_name]["f1_score"]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with F1: {best_model_score:.4f}")

            if best_model_score < 0.75:
                raise CustomException("No model achieved acceptable F1 score (minimum 0.75)", sys)

            # --- Log best model to MLflow ---
            with mlflow.start_run(run_name=best_model_name):
                mlflow.log_param("model_name", best_model_name)
                mlflow.log_params(model_report[best_model_name]["best_params"])
                mlflow.log_metric("f1_score", model_report[best_model_name]["f1_score"])
                mlflow.log_metric("recall", model_report[best_model_name]["recall"])
                mlflow.log_metric("precision", model_report[best_model_name]["precision"])
                mlflow.log_metric("roc_auc", model_report[best_model_name]["roc_auc"])
                mlflow.log_metric("pr_auc", model_report[best_model_name]["pr_auc"])
                mlflow.sklearn.log_model(best_model, "model")

            logging.info("Best model logged to MLflow/DagsHub")

            # --- Save model locally ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            # --- Final report ---
            y_pred = best_model.predict(X_test)
            print(f"\nBest Model: {best_model_name}")
            print(classification_report(y_test, y_pred, target_names=["False Positive", "Confirmed"]))

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import pandas as pd
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _ = transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    model_path = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
    print("Model saved at:", model_path)