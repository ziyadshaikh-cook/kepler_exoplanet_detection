import sys
import pandas as pd
from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData
from src.exception import CustomException

app = Flask(__name__)

# Median values for the 26 fields not shown in the form
MEDIAN_DEFAULTS = {
    "koi_period_err1": 0.000022,
    "koi_period_err2": -0.000022,
    "koi_time0bk": 170.54,
    "koi_time0bk_err1": 0.0013,
    "koi_time0bk_err2": -0.0013,
    "koi_impact_err1": 0.19,
    "koi_impact_err2": -0.19,
    "koi_duration_err1": 0.021,
    "koi_duration_err2": -0.021,
    "koi_depth_err1": 29.0,
    "koi_depth_err2": -29.0,
    "koi_prad_err1": 0.16,
    "koi_prad_err2": -0.16,
    "koi_teq": 793.0,
    "koi_insol_err1": 7.65,
    "koi_insol_err2": -7.65,
    "koi_tce_plnt_num": 1.0,
    "koi_steff_err1": 81.0,
    "koi_steff_err2": -81.0,
    "koi_slogg_err1": 0.071,
    "koi_slogg_err2": -0.096,
    "koi_srad_err1": 0.105,
    "koi_srad_err2": -0.061,
    "koi_kepmag": 14.48,
    "koi_insol": 176.5,
    "koi_model_snr": 35.8,
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        data = CustomData(
            koi_fpflag_nt=float(form.get("koi_fpflag_nt", 0)),
            koi_fpflag_ss=float(form.get("koi_fpflag_ss", 0)),
            koi_fpflag_co=float(form.get("koi_fpflag_co", 0)),
            koi_fpflag_ec=float(form.get("koi_fpflag_ec", 0)),
            koi_period=float(form["koi_period"]),
            koi_period_err1=MEDIAN_DEFAULTS["koi_period_err1"],
            koi_period_err2=MEDIAN_DEFAULTS["koi_period_err2"],
            koi_time0bk=MEDIAN_DEFAULTS["koi_time0bk"],
            koi_time0bk_err1=MEDIAN_DEFAULTS["koi_time0bk_err1"],
            koi_time0bk_err2=MEDIAN_DEFAULTS["koi_time0bk_err2"],
            koi_impact=float(form["koi_impact"]),
            koi_impact_err1=MEDIAN_DEFAULTS["koi_impact_err1"],
            koi_impact_err2=MEDIAN_DEFAULTS["koi_impact_err2"],
            koi_duration=float(form["koi_duration"]),
            koi_duration_err1=MEDIAN_DEFAULTS["koi_duration_err1"],
            koi_duration_err2=MEDIAN_DEFAULTS["koi_duration_err2"],
            koi_depth=float(form["koi_depth"]),
            koi_depth_err1=MEDIAN_DEFAULTS["koi_depth_err1"],
            koi_depth_err2=MEDIAN_DEFAULTS["koi_depth_err2"],
            koi_prad=float(form["koi_prad"]),
            koi_prad_err1=MEDIAN_DEFAULTS["koi_prad_err1"],
            koi_prad_err2=MEDIAN_DEFAULTS["koi_prad_err2"],
            koi_teq=MEDIAN_DEFAULTS["koi_teq"],
            koi_insol=MEDIAN_DEFAULTS["koi_insol"],
            koi_insol_err1=MEDIAN_DEFAULTS["koi_insol_err1"],
            koi_insol_err2=MEDIAN_DEFAULTS["koi_insol_err2"],
            koi_model_snr=float(form["koi_model_snr"]),
            koi_tce_plnt_num=MEDIAN_DEFAULTS["koi_tce_plnt_num"],
            koi_steff=float(form["koi_steff"]),
            koi_steff_err1=MEDIAN_DEFAULTS["koi_steff_err1"],
            koi_steff_err2=MEDIAN_DEFAULTS["koi_steff_err2"],
            koi_slogg=float(form["koi_slogg"]),
            koi_slogg_err1=MEDIAN_DEFAULTS["koi_slogg_err1"],
            koi_slogg_err2=MEDIAN_DEFAULTS["koi_slogg_err2"],
            koi_srad=float(form["koi_srad"]),
            koi_srad_err1=MEDIAN_DEFAULTS["koi_srad_err1"],
            koi_srad_err2=MEDIAN_DEFAULTS["koi_srad_err2"],
            koi_kepmag=MEDIAN_DEFAULTS["koi_kepmag"],
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictionPipeline()
        prediction, probability = pipeline.predict(df)

        result = "Confirmed Exoplanet" if prediction[0] == 1 else "False Positive"
        confidence = round(float(probability[0]) * 100, 2)

        return render_template("results.html", result=result, confidence=confidence)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)