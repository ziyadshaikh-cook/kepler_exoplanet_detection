# Kepler Exoplanet Detection

An end-to-end machine learning pipeline to classify celestial objects as confirmed exoplanets or false positives using real NASA Kepler Space Telescope observation data.

***

## Problem Statement

The NASA Kepler Space Telescope observed hundreds of thousands of stars looking for periodic dips in brightness — a signal that a planet might be passing in front of the star. Not all signals are real planets. This project builds a binary classifier to distinguish **confirmed exoplanets** from **false positives** using the telescope's recorded measurements.

This is a severe class imbalance problem. Only ~30% of Kepler Objects of Interest (KOIs) are confirmed exoplanets. A naive model that predicts "not an exoplanet" for everything would still get high accuracy — which is why F1-score and Recall are the primary evaluation metrics here, not accuracy.

***

## Dataset

- **Source:** [NASA Kepler Exoplanet Search Results — Kaggle](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)
- **Target column:** `koi_disposition` → encoded as `CONFIRMED = 1`, everything else = `0`
- **Features:** ~50 continuous numeric measurements — orbital period, transit duration, planet radius estimates, stellar temperature, surface gravity, signal-to-noise ratios
- **Key challenge:** ID/metadata columns (`kepid`, `kepoi_name`, `kepler_name`, `koi_pdisposition`) must be stripped before the model sees any data. These columns directly encode the answer and would cause severe data leakage.

***

## Pipeline Architecture

```
Raw CSV (Kaggle)
      │
      ▼
data_ingestion.py ──────────► artifacts/train.csv + test.csv
      │
      ▼
data_transformation.py ──────► artifacts/preprocessor.pkl
  (drop ID cols, impute NaNs,
   RobustScaler, SMOTE on train)
      │
      ▼
model_trainer.py ────────────► artifacts/model.pkl
  (XGBoost, RF, GB, LR)          (MLflow + DagsHub tracking)
  (metric: F1 + Recall)
      │
      ▼
training_pipeline.py ────────► chains all 3 above via main.py
      │
      ▼
prediction_pipeline.py ──────► loads preprocessor + model, runs inference
      │
      ▼
app.py (Flask) ───────────────► web UI for single-object prediction
      │
      ▼
model_monitoring.py ─────────► Evidently data drift report
      │
      ▼
Dockerfile ───────────────────► containerized deployment
```

***

## Project Structure

```
kepler_exoplanet_detection/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_monitoring.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   ├── prediction_pipeline.py
│   │   └── monitoring_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── notebook/
│   ├── 01_EDA.ipynb
│   └── 02_model_experiment.ipynb
├── templates/
│   ├── index.html
│   └── results.html
├── data/raw/          ← place Kaggle CSV here
├── artifacts/         ← generated at runtime (gitignored)
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
└── setup.py
```

***

## Tech Stack

| Component | Library |
|---|---|
| ML Models | scikit-learn, XGBoost |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow + DagsHub |
| Web App | Flask |
| Monitoring | Evidently |
| Containerization | Docker |
| Environment | Python 3.10, conda/venv |

***

## Key Engineering Decisions

**Why RobustScaler over StandardScaler?**
Astronomy measurements contain legitimate outliers — an unusually high signal-to-noise ratio is real signal, not noise. RobustScaler uses median and IQR instead of mean and std, so extreme values don't distort the scaling for the rest of the features.

**Why SMOTE on training data only?**
SMOTE generates synthetic minority class samples. It is applied only to `X_train` after the train/test split. Applying it before splitting would leak synthetic samples into the test set, giving falsely optimistic evaluation metrics.

**Why F1-score as the primary metric?**
With class imbalance, accuracy is misleading. A model predicting "not an exoplanet" for every object scores high accuracy but detects zero planets. F1-score balances precision and recall. Recall specifically matters here — missing a real exoplanet is worse than a false alarm.

**Why drop CANDIDATE rows?**
CANDIDATE means the object has not been confirmed or ruled out — it is genuinely ambiguous. Including it as a negative label would add noise to the model. The binary decision was: CONFIRMED = 1, FALSE POSITIVE = 0, CANDIDATE dropped.

***

## How to Run Locally

**1. Clone the repo and set up environment**
```bash
git clone https://github.com/ziyadshaikh-cook/kepler_exoplanet_detection.git
cd kepler_exoplanet_detection
pip install -r requirements.txt
```

**2. Place the dataset**

Download `cumulative.csv` from [Kaggle](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) and place it in `data/raw/`.

**3. Run the training pipeline**
```bash
python main.py
```

This runs data ingestion → transformation → model training end to end. Artifacts are saved to `artifacts/`.

**4. Run the Flask app**
```bash
python app.py
```

Navigate to `http://localhost:5000` to use the prediction UI.

**5. Run the monitoring report**
```bash
python src/pipeline/monitoring_pipeline.py
```

Report saved to `artifacts/monitoring_report.html`.

***

## How to Run with Docker

```bash
docker build -t kepler-exoplanet .
docker run -p 5000:5000 kepler-exoplanet
```

Navigate to `http://localhost:5000`.

> Note: The Docker image does not include trained artifacts. Run `python main.py` first to generate `artifacts/model.pkl` and `artifacts/preprocessor.pkl`, then build the image.

***

## MLflow / DagsHub Tracking

Experiments are tracked on [DagsHub](https://dagshub.com/ziyadshaikh-cook/kepler_exoplanet_detection).

Metrics logged per model run:
- F1-score
- Recall
- Precision
- ROC-AUC
- Precision-Recall AUC

***

## What's Different vs My Churn Prediction Project

| Dimension | Customer Churn | Exoplanet Detection |
|---|---|---|
| Domain | Business/customer data | Scientific telescope measurements |
| Features | ~20, mixed categorical + numeric | ~50, almost all continuous numeric |
| Class imbalance | Moderate | Severe — required SMOTE |
| Scaler | StandardScaler | RobustScaler (outlier-robust) |
| Primary metric | Accuracy / AUC | F1-score + Recall |
| Data leakage risk | Encoding | ID columns encode the answer directly |
| Feature engineering | Ordinal encoding | Ratio features, drop error uncertainty cols |

***

## Author

**Ziyad Shaikh**
[GitHub](https://github.com/ziyadshaikh-cook)