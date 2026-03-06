import pandas as pd
import joblib

# load model
model = joblib.load("energy_model.pkl")
features = joblib.load("model_features.pkl")

# contoh input baru
sample_input = pd.DataFrame([{
    "lag_1": 640000,
    "day_of_week": 2,
    "rolling_mean_7": 650000,
    "rolling_std_7": 20000,
    "lag_7": 660000
}])

sample_input = sample_input[features]

prediction = model.predict(sample_input)

print("Predicted energy:", prediction[0])