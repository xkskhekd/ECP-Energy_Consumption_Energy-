import streamlit as st
import matplotlib.pyplot as plt
import sys
import pandas as pd
import json
from matplotlib.ticker import FuncFormatter

sys.path.append("src")

from forecast import generate_forecast

st.title("Energy Consumption Forecast Dashboard")

with open("models/model_metrics.json", "r") as f:metrics = json.load(f)

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metrics['MAE']:.0f}")
col2.metric("RMSE", f"{metrics['RMSE']:.0f}")
col3.metric("R² Score", f"{metrics['R2']:.2f}")

st.write("Predict future energy consumption using machine learning.")

# dataset information

df_info = pd.read_csv("dataset/PJME_hourly.csv")

start_data = pd.to_datetime(df_info.iloc[0,0])
end_data = pd.to_datetime(df_info.iloc[-1,0])

colA, colB, colC = st.columns(3)

colA.metric("Dataset Start", start_data.strftime("%Y-%m-%d"))
colB.metric("Dataset End", end_data.strftime("%Y-%m-%d"))
colC.metric("Total Rows", len(df_info))

with open("models/feature_importance.json", "r") as f:importance = json.load(f)

imp_df = pd.DataFrame(
list(importance.items()),
columns=["feature", "importance"]
).sort_values("importance", ascending=False)

st.subheader("Model Feature Importance")

fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.barh(imp_df["feature"], imp_df["importance"])
ax2.invert_yaxis()

ax2.set_xlabel("Importance Score")
ax2.set_ylabel("Feature")

st.pyplot(fig2)

eval_df = pd.read_csv("models/model_evaluation.csv")

eval_df["date"] = pd.to_datetime(eval_df["date"])

eval_df = eval_df.sort_values("date")

st.subheader("Model Evaluation: Actual vs Prediction")

fig_eval, ax_eval = plt.subplots(figsize=(12,5))

recent_eval = eval_df.tail(365)

ax_eval.plot(
recent_eval["date"],
recent_eval["actual"],
label="Actual",
color="blue"
)

ax_eval.plot(
recent_eval["date"],
recent_eval["predicted"],
label="Predicted",
color="orange"
)

ax_eval.set_xlabel("Date")
ax_eval.set_ylabel("Energy Consumption")
ax_eval.set_title("Model Prediction vs Actual")

ax_eval.legend()

st.pyplot(fig_eval)

@st.cache_data
def run_forecast(start, end):return generate_forecast(start, end)

df = pd.read_csv("dataset/PJME_hourly.csv")

df.columns = ["datetime", "energy"]

df["datetime"] = pd.to_datetime(df["datetime"])

last_date = df["datetime"].max()

min_date = (last_date + pd.Timedelta(days=1)).date()

start_date = st.date_input(
    "Forecast start date",
    value=min_date,
    min_value=min_date
)

default_end = start_date + pd.Timedelta(days=7)

end_date = st.date_input(
    "Forecast end date",
    value=default_end,
    min_value=start_date
)

if st.button("Run Forecast"):

    try:
        forecast_df, history_df = run_forecast(start_date, end_date)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

    st.subheader("Forecast Result")
    st.dataframe(forecast_df)

    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name="energy_forecast.csv",
        mime="text/csv"
    )

    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        forecast_df["date"],
        forecast_df["predicted_energy"],
        label="Forecast",
        color="orange"
    )

    ax.axvline(
        pd.Timestamp(start_date),
        linestyle="--",
        color="red",
        label="Forecast Start"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Consumption")
    ax.set_title("Energy Consumption Forecast")

    ax.legend()

    st.pyplot(fig)
