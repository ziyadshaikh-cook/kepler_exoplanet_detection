import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class PredictionPipelineConfig:
    model_path: str = "artifacts/model.pkl"
    preprocessor_path: str = "artifacts/preprocessor.pkl"


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading model and preprocessor for prediction")
            model = load_object(self.config.model_path)
            preprocessor = load_object(self.config.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)[:, 1]

            logging.info(f"Prediction: {prediction[0]} | Probability: {probability[0]:.4f}")
            return prediction, probability

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps raw input from the Flask form into a DataFrame
    that matches the feature columns the model was trained on.
    """
    def __init__(self,
                 koi_fpflag_nt: float,
                 koi_fpflag_ss: float,
                 koi_fpflag_co: float,
                 koi_fpflag_ec: float,
                 koi_period: float,
                 koi_period_err1: float,
                 koi_period_err2: float,
                 koi_time0bk: float,
                 koi_time0bk_err1: float,
                 koi_time0bk_err2: float,
                 koi_impact: float,
                 koi_impact_err1: float,
                 koi_impact_err2: float,
                 koi_duration: float,
                 koi_duration_err1: float,
                 koi_duration_err2: float,
                 koi_depth: float,
                 koi_depth_err1: float,
                 koi_depth_err2: float,
                 koi_prad: float,
                 koi_prad_err1: float,
                 koi_prad_err2: float,
                 koi_teq: float,
                 koi_insol: float,
                 koi_insol_err1: float,
                 koi_insol_err2: float,
                 koi_model_snr: float,
                 koi_tce_plnt_num: float,
                 koi_steff: float,
                 koi_steff_err1: float,
                 koi_steff_err2: float,
                 koi_slogg: float,
                 koi_slogg_err1: float,
                 koi_slogg_err2: float,
                 koi_srad: float,
                 koi_srad_err1: float,
                 koi_srad_err2: float,
                 koi_kepmag: float):

        self.koi_fpflag_nt = koi_fpflag_nt
        self.koi_fpflag_ss = koi_fpflag_ss
        self.koi_fpflag_co = koi_fpflag_co
        self.koi_fpflag_ec = koi_fpflag_ec
        self.koi_period = koi_period
        self.koi_period_err1 = koi_period_err1
        self.koi_period_err2 = koi_period_err2
        self.koi_time0bk = koi_time0bk
        self.koi_time0bk_err1 = koi_time0bk_err1
        self.koi_time0bk_err2 = koi_time0bk_err2
        self.koi_impact = koi_impact
        self.koi_impact_err1 = koi_impact_err1
        self.koi_impact_err2 = koi_impact_err2
        self.koi_duration = koi_duration
        self.koi_duration_err1 = koi_duration_err1
        self.koi_duration_err2 = koi_duration_err2
        self.koi_depth = koi_depth
        self.koi_depth_err1 = koi_depth_err1
        self.koi_depth_err2 = koi_depth_err2
        self.koi_prad = koi_prad
        self.koi_prad_err1 = koi_prad_err1
        self.koi_prad_err2 = koi_prad_err2
        self.koi_teq = koi_teq
        self.koi_insol = koi_insol
        self.koi_insol_err1 = koi_insol_err1
        self.koi_insol_err2 = koi_insol_err2
        self.koi_model_snr = koi_model_snr
        self.koi_tce_plnt_num = koi_tce_plnt_num
        self.koi_steff = koi_steff
        self.koi_steff_err1 = koi_steff_err1
        self.koi_steff_err2 = koi_steff_err2
        self.koi_slogg = koi_slogg
        self.koi_slogg_err1 = koi_slogg_err1
        self.koi_slogg_err2 = koi_slogg_err2
        self.koi_srad = koi_srad
        self.koi_srad_err1 = koi_srad_err1
        self.koi_srad_err2 = koi_srad_err2
        self.koi_kepmag = koi_kepmag

    def get_data_as_dataframe(self):
        try:
            data = {
                "koi_fpflag_nt": [self.koi_fpflag_nt],
                "koi_fpflag_ss": [self.koi_fpflag_ss],
                "koi_fpflag_co": [self.koi_fpflag_co],
                "koi_fpflag_ec": [self.koi_fpflag_ec],
                "koi_period": [self.koi_period],
                "koi_period_err1": [self.koi_period_err1],
                "koi_period_err2": [self.koi_period_err2],
                "koi_time0bk": [self.koi_time0bk],
                "koi_time0bk_err1": [self.koi_time0bk_err1],
                "koi_time0bk_err2": [self.koi_time0bk_err2],
                "koi_impact": [self.koi_impact],
                "koi_impact_err1": [self.koi_impact_err1],
                "koi_impact_err2": [self.koi_impact_err2],
                "koi_duration": [self.koi_duration],
                "koi_duration_err1": [self.koi_duration_err1],
                "koi_duration_err2": [self.koi_duration_err2],
                "koi_depth": [self.koi_depth],
                "koi_depth_err1": [self.koi_depth_err1],
                "koi_depth_err2": [self.koi_depth_err2],
                "koi_prad": [self.koi_prad],
                "koi_prad_err1": [self.koi_prad_err1],
                "koi_prad_err2": [self.koi_prad_err2],
                "koi_teq": [self.koi_teq],
                "koi_insol": [self.koi_insol],
                "koi_insol_err1": [self.koi_insol_err1],
                "koi_insol_err2": [self.koi_insol_err2],
                "koi_model_snr": [self.koi_model_snr],
                "koi_tce_plnt_num": [self.koi_tce_plnt_num],
                "koi_steff": [self.koi_steff],
                "koi_steff_err1": [self.koi_steff_err1],
                "koi_steff_err2": [self.koi_steff_err2],
                "koi_slogg": [self.koi_slogg],
                "koi_slogg_err1": [self.koi_slogg_err1],
                "koi_slogg_err2": [self.koi_slogg_err2],
                "koi_srad": [self.koi_srad],
                "koi_srad_err1": [self.koi_srad_err1],
                "koi_srad_err2": [self.koi_srad_err2],
                "koi_kepmag": [self.koi_kepmag],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)