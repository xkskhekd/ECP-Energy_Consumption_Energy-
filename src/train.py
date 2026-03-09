import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("dataset/PJME_hourly.csv")
df.columns = ["datetime", "energy"]

df["datetime"] = pd.to_datetime(df["datetime"])

print("Dataset loaded")
print("Rows:", len(df))


# =========================
# SORT TIMESTAMP
# =========================

df = df.sort_values("datetime")


# =========================
# HANDLE DUPLICATE DST HOURS
# =========================

df = df.groupby("datetime", as_index=False)["energy"].mean()


# =========================
# ENSURE HOURLY CONTINUITY
# =========================

df = df.set_index("datetime")

df = df.resample("h").mean()

df["energy"] = df["energy"].interpolate()

df = df.reset_index()

print("After cleaning rows:", len(df))


# =========================
# RESAMPLE HOURLY → DAILY
# =========================

daily_df = df.resample("D", on="datetime").sum()

print("Daily rows:", len(daily_df))


# =========================
# FEATURE ENGINEERING
# =========================

daily_df["day_of_week"] = daily_df.index.dayofweek
daily_df["month"] = daily_df.index.month

# cyclical encoding
daily_df["dow_sin"] = np.sin(2 * np.pi * daily_df["day_of_week"] / 7)
daily_df["dow_cos"] = np.cos(2 * np.pi * daily_df["day_of_week"] / 7)

daily_df["month_sin"] = np.sin(2 * np.pi * daily_df["month"] / 12)
daily_df["month_cos"] = np.cos(2 * np.pi * daily_df["month"] / 12)


# =========================
# LAG FEATURES
# =========================

daily_df["lag_1"] = daily_df["energy"].shift(1)
daily_df["lag_7"] = daily_df["energy"].shift(7)
daily_df["lag_14"] = daily_df["energy"].shift(14)
daily_df["lag_21"] = daily_df["energy"].shift(21)
daily_df["lag_28"] = daily_df["energy"].shift(28)
daily_df["lag_30"] = daily_df["energy"].shift(30)
daily_df["lag_365"] = daily_df["energy"].shift(365)


# =========================
# ROLLING FEATURES
# =========================

daily_df["rolling_mean_7"] = daily_df["energy"].shift(1).rolling(7).mean()
daily_df["rolling_mean_14"] = daily_df["energy"].shift(1).rolling(14).mean()


daily_df = daily_df.dropna()

print("Final dataset size:", daily_df.shape)


# =========================
# FEATURE SELECTION
# =========================

features = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_30",
    "lag_365",
    "rolling_mean_7",
    "rolling_mean_14",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos"
]

X = daily_df[features]
y = daily_df["energy"]


# =========================
# TIME SERIES CROSS VALIDATION
# =========================

tscv = TimeSeriesSplit(n_splits=5)

mae_scores = []
rmse_scores = []
r2_scores = []

for train_index, test_index in tscv.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = XGBRegressor(
        n_estimators=900,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)


print("\nCross Validation Results")
print("Average MAE:", np.mean(mae_scores))
print("Average RMSE:", np.mean(rmse_scores))
print("Average R2:", np.mean(r2_scores))


# =========================
# TRAIN FINAL MODEL
# =========================

model.fit(X, y)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/energy_model.pkl")
joblib.dump(features, "models/model_features.pkl")

print("\nModel saved")


# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(15,5))

plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Prediction")

plt.legend()
plt.title("Energy Consumption Prediction")

plt.show()