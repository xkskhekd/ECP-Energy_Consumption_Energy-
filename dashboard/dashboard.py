import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecast import generate_forecast

st.title("Energy Consumption Forecast")

start_date = st.date_input("Start date")
end_date = st.date_input("End date")

if st.button("Run Forecast"):

    forecast_df, history_df = generate_forecast(start_date, end_date)

    st.write(forecast_df)

    fig, ax = plt.subplots()

    ax.plot(history_df.index[-120:], history_df["energy"][-120:], label="Historical")
    ax.plot(forecast_df["date"], forecast_df["predicted_energy"], label="Forecast")

    ax.legend()

    st.pyplot(fig)