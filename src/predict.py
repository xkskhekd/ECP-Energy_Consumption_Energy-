import pandas as pd
import joblib

# load model
model = joblib.load("models/energy_model.pkl")
features = joblib.load("models/model_features.pkl")

# contoh input
data = {
    "lag_1": 640000,
    "day_of_week": 2,
    "rolling_mean_7": 650000,
    "rolling_std_7": 20000,
    "lag_7": 660000
}

sample_input = pd.DataFrame([data])

# pastikan urutan fitur sama
sample_input = sample_input[features]

prediction = model.predict(sample_input)

print(f"Predicted energy: {prediction[0]:,.2f}")