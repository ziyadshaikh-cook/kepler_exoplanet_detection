import os
import sys
import dill
import numpy as np
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Hyperparameter tuning
            gs = GridSearchCV(model, params[model_name], cv=3, scoring="f1", n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else None

            report[model_name] = {
                "f1_score": f1,
                "recall": recall,
                "precision": precision,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "best_params": gs.best_params_
            }

            logging.info(f"{model_name} -> F1: {f1:.4f} | Recall: {recall:.4f} | PR-AUC: {pr_auc:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)