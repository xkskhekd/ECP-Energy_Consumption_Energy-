# Energy Consumption Forecast

Small machine learning project to predict future electricity consumption based on historical usage data.

Model menggunakan **XGBoost regression** dengan pendekatan **time-series forecasting**.
Hasil training dan forecasting dapat dilihat melalui **Streamlit dashboard**.

Project ini dibuat untuk latihan membangun pipeline ML sederhana dari data → model → visualization.

---

## Project Structure

```
energy-consumption-forecast/
│
├── dataset/
│   └── PJME_hourly.csv
│
├── models/
│   ├── energy_model.pkl
│   ├── model_features.pkl
│   ├── model_metrics.json
│   ├── feature_importance.json
│   └── model_evaluation.csv
│
├── src/
│   ├── train.py
│   └── forecast.py
│
└── dashboard/
    └── dashboard.py
```

---

## Features

Model menggunakan beberapa fitur time-series sederhana:

* lag features (1, 7, 14, 21, 28, 30, 365)
* rolling mean (7 dan 14 hari)
* cyclical encoding untuk day-of-week dan month

Tujuannya agar model bisa menangkap **weekly pattern** dan **seasonal behaviour** dari konsumsi energi.

---

## Training

Untuk melatih model:

```
python src/train.py
```

Script ini akan:

* load dataset
* preprocessing data
* generate features
* train model menggunakan TimeSeriesSplit
* menyimpan model ke folder `models/`

---

## Dashboard

Dashboard dibuat menggunakan **Streamlit**.

Jalankan dengan:

```
streamlit run dashboard/dashboard.py
```

Di dashboard kamu bisa:

* melihat model metrics
* melihat feature importance
* membandingkan actual vs predicted
* menjalankan forecast
* export hasil forecast ke CSV

---

## Dependencies

Library utama yang digunakan:

```
pandas
numpy
xgboost
scikit-learn
matplotlib
streamlit
joblib
```

Install dengan:

```
pip install -r requirements.txt
```

---

## Notes

Project ini merupakan implementasi sederhana untuk memahami workflow forecasting dengan machine learning.
Masih banyak ruang untuk pengembangan seperti hyperparameter tuning, experiment tracking, atau deployment sebagai API.
