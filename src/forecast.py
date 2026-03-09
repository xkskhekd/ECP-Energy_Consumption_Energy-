import pandas as pd
import numpy as np
import joblib
import sys

# ======================
# READ ARGUMENTS
# ======================

if len(sys.argv) != 3:
    print("Usage: python forecast.py START_DATE END_DATE")
    print("Example: python forecast.py 2026-03-30 2026-04-30")
    sys.exit()

start_date = pd.Timestamp(sys.argv[1])
end_date = pd.Timestamp(sys.argv[2])

forecast_days = (end_date - start_date).days + 1

print("Forecast from:", start_date)
print("Forecast until:", end_date)
print("Total days:", forecast_days)


# ======================
# LOAD MODEL
# ======================

model = joblib.load("models/energy_model.pkl")
features = joblib.load("models/model_features.pkl")


# ======================
# LOAD DATA
# ======================

df = pd.read_csv("dataset/PJME_hourly.csv")
df.columns = ["datetime", "energy"]

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime")
df = df.groupby("datetime", as_index=False)["energy"].mean()

df = df.set_index("datetime")
df = df.resample("h").mean()
df["energy"] = df["energy"].interpolate()

df = df.reset_index()


# ======================
# DAILY DATA
# ======================

daily_df = df.resample("D", on="datetime").sum()


# ======================
# FORECASTING
# ======================

history = daily_df.copy()

predictions = []

for i in range(forecast_days):

    next_date = start_date + pd.Timedelta(days=i)

    day_of_week = next_date.dayofweek
    month = next_date.month

    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    lag_1 = history["energy"].iloc[-1]
    lag_7 = history["energy"].iloc[-7]
    lag_14 = history["energy"].iloc[-14]
    lag_21 = history["energy"].iloc[-21]
    lag_28 = history["energy"].iloc[-28]
    lag_30 = history["energy"].iloc[-30]
    lag_365 = history["energy"].iloc[-365]

    rolling_mean_7 = history["energy"].iloc[-7:].mean()
    rolling_mean_14 = history["energy"].iloc[-14:].mean()

    X = pd.DataFrame([[
        lag_1,
        lag_7,
        lag_14,
        lag_21,
        lag_28,
        lag_30,
        lag_365,
        rolling_mean_7,
        rolling_mean_14,
        dow_sin,
        dow_cos,
        month_sin,
        month_cos
    ]], columns=features)

    pred = model.predict(X)[0]

    predictions.append((next_date, pred))

    history.loc[next_date] = pred


# ======================
# RESULT
# ======================

forecast_df = pd.DataFrame(predictions, columns=["date", "predicted_energy"])

print("\nForecast result\n")
print(forecast_df)

forecast_df.to_csv("dataset/forecast_custom_range.csv", index=False)

print("\nForecast saved to forecast_custom_range.csv")