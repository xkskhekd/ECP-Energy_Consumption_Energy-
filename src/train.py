import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load dataset
df = pd.read_csv("dataset/PJME_hourly.csv")
df.columns = ["datetime", "energy"]

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# ubah ke daily
daily_df = df.resample("D", on="datetime").sum()

# datetime features
daily_df["day_of_week"] = daily_df.index.dayofweek
daily_df["month"] = daily_df.index.month
daily_df["day_of_year"] = daily_df.index.dayofyear
daily_df["week_of_year"] = daily_df.index.isocalendar().week.astype(int)
daily_df["quarter"] = daily_df.index.quarter

# lag features
daily_df["lag_1"] = daily_df["energy"].shift(1)
daily_df["lag_7"] = daily_df["energy"].shift(7)
daily_df["lag_14"] = daily_df["energy"].shift(14)
daily_df["lag_30"] = daily_df["energy"].shift(30)

# rolling statistics
daily_df["rolling_mean_7"] = daily_df["energy"].rolling(window=7).mean()
daily_df["rolling_std_7"] = daily_df["energy"].rolling(window=7).std()
daily_df["rolling_mean_14"] = daily_df["energy"].rolling(window=14).mean()
daily_df["rolling_std_14"] = daily_df["energy"].rolling(window=14).std()

daily_df = daily_df.dropna()

# pilih fitur terbaik
X = daily_df[[
    "lag_1",
    "day_of_week",
    "rolling_mean_7",
    "rolling_std_7",
    "lag_7"
]]

y = daily_df["energy"]

# split data
train_size = int(len(daily_df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print(X_train.shape)
print(X_test.shape)

# model
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# training
model.fit(X_train, y_train)

# simpan model
joblib.dump(model, "energy_model.pkl")

# simpan fitur
features = X_train.columns.tolist()
joblib.dump(features, "model_features.pkl")

# evaluasi
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# plot hasil
plt.figure(figsize=(15,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Prediction")
plt.legend()
plt.title("Energy Consumption Prediction")
plt.show()