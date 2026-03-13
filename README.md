# Energy Consumption Forecast

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Active-success)

Machine learning system for forecasting electricity consumption using time-series feature engineering and gradient boosting models.
The project includes a complete ML pipeline from training to interactive forecasting via a web dashboard.

---

# Key Features

• Time-series forecasting pipeline
• Feature engineering for seasonal patterns
• Model training with time-series cross-validation
• Forecast generation for future dates
• Interactive Streamlit dashboard
• Exportable forecast results

---

# Tech Stack

Python
Pandas
XGBoost
Scikit-learn
Matplotlib
Streamlit

---

# Machine Learning Pipeline

Raw dataset → Cleaning → Feature Engineering → Model Training → Evaluation → Forecast Engine → Dashboard

Key feature types:

Lag features
Rolling statistics
Cyclical time encoding

Evaluation metrics:

MAE
RMSE
R² Score

---

# Dashboard

The dashboard allows users to:

• view model metrics
• inspect feature importance
• evaluate predictions vs actual data
• generate future forecasts
• export forecast results as CSV

Run dashboard:

```
streamlit run dashboard/dashboard.py
```

---

# Forecast Example

Users choose:

Forecast start date
Forecast end date

The system generates predicted energy consumption for the selected range.

---

# Project Structure

```
energy-consumption-forecast/

dataset/
PJME_hourly.csv

models/
energy_model.pkl
model_features.pkl
model_metrics.json
model_evaluation.csv
feature_importance.json
train_config_used.json

config/
train_config.json

src/
train.py
forecast.py

dashboard/
dashboard.py

requirements.txt
README.md
```

---

# Train Model

```
python src/train.py
```

Artifacts generated:

trained model
feature importance
evaluation dataset
model metrics

---

# Future Improvements

• hyperparameter tuning pipeline
• experiment tracking
• API-based forecasting service
• Docker deployment
• automated retraining pipeline

---

# Author

Machine Learning Portfolio Project

Focus areas:

Energy Systems
Machine Learning Engineering
Time Series Forecasting

![Stars](https://img.shields.io/github/stars/username/repo)
![Forks](https://img.shields.io/github/forks/username/repo)