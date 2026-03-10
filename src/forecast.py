import pandas as pd
import numpy as np
import joblib

def generate_forecast(start_date, end_date):

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    forecast_days = (end_date - start_date).days + 1

    model = joblib.load("models/energy_model.pkl")
    features = joblib.load("models/model_features.pkl")

    df = pd.read_csv("dataset/PJME_hourly.csv")
    df.columns = ["datetime", "energy"]

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime")
    df = df.groupby("datetime", as_index=False)["energy"].mean()

    df = df.set_index("datetime")
    df = df.resample("h").mean()

    df["energy"] = df["energy"].interpolate()
    df["energy"] = df["energy"].ffill()
    df["energy"] = df["energy"].bfill()

    df = df.reset_index()

    daily_df = df.resample("D", on="datetime").sum()

    # hapus hari terakhir jika tidak lengkap
    daily_df = daily_df.iloc[:-1]

    history = daily_df.copy()

    predictions = []

    for i in range(forecast_days):

        next_date = start_date + pd.Timedelta(days=i)

        dow = next_date.dayofweek
        month = next_date.month

        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

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

    forecast_df = pd.DataFrame(predictions, columns=["date","predicted_energy"])

    return forecast_df, daily_df