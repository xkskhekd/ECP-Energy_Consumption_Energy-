Energy Consumption Forecast

Simple machine learning project to predict future electricity consumption based on historical usage data.
Model dilatih menggunakan gradient boosted trees (XGBoost) dengan pendekatan time-series forecasting dan divisualisasikan melalui Streamlit dashboard.

Project ini dibuat sebagai latihan membangun pipeline ML end-to-end: mulai dari data preprocessing, feature engineering, training model, sampai deployment sederhana lewat dashboard.

Project Structure
energy-consumption-forecast/

dataset/
    PJME_hourly.csv

models/
    energy_model.pkl
    model_features.pkl
    model_metrics.json
    feature_importance.json
    model_evaluation.csv

src/
    train.py
    forecast.py

dashboard/
    dashboard.py

dataset/
Raw hourly electricity consumption data.

models/
Model artifacts hasil training:

trained model

feature list

evaluation metrics

feature importance

prediction results

src/
Core machine learning scripts:

training pipeline

forecasting logic

dashboard/
Streamlit dashboard untuk melihat performa model dan menjalankan forecast.

Features

Beberapa fitur time-series yang digunakan:

Lag features (lag_1, lag_7, lag_14, dst)

Rolling statistics (7-day dan 14-day mean)

Cyclical time encoding untuk day-of-week dan month

Pendekatan ini membantu model menangkap weekly pattern dan seasonality pada konsumsi energi.

Model

Model yang digunakan:

XGBoost Regressor

Training dilakukan dengan TimeSeriesSplit cross validation agar urutan waktu tetap terjaga.

Metrics yang digunakan:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

Training

Untuk melatih model:

python src/train.py

Script ini akan:

load dan clean dataset

generate features

train model

evaluate performa

menyimpan model ke folder models/

Dashboard

Dashboard dibuat menggunakan Streamlit untuk melihat hasil model dan menjalankan prediksi.

Jalankan dengan:

streamlit run dashboard/dashboard.py

Dashboard menampilkan:

model metrics

feature importance

actual vs predicted plot

energy consumption forecast

export forecast ke CSV

Requirements

Beberapa dependency utama:

pandas
numpy
xgboost
scikit-learn
matplotlib
streamlit
joblib

Install dengan:

pip install -r requirements.txt
Notes

Project ini dibuat sebagai learning project untuk memahami pipeline forecasting sederhana menggunakan machine learning.

Masih banyak kemungkinan pengembangan, misalnya:

hyperparameter tuning

tambahan time-series features

experiment tracking

model comparison

API untuk inference