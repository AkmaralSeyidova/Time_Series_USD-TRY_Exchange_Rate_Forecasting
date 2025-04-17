import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

st.title("USD/TRY Exchange Rate Forecasting with ARIMA")

uploaded_file = st.file_uploader("Upload your USD/TRY CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df = df.asfreq("D")
    df = df.fillna(method="ffill")

    st.subheader("USD/TRY Time Series")
    st.line_chart(df["USD_TRY"])

    # Train/Test Split
    train = df[:'2022-12-31']
    test = df['2023-01-01':]

    # Fit ARIMA
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))
    test["Predicted"] = forecast.values

    # Show plot
    st.subheader("Forecast vs Actual")
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(train.index, train["USD_TRY"], label="Train")
    ax.plot(test.index, test["USD_TRY"], label="Test")
    ax.plot(test.index, test["Predicted"], label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # RMSE
    rmse = sqrt(mean_squared_error(test["USD_TRY"], test["Predicted"]))
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
else:
    st.info("Please upload a CSV file with 'Date' and 'USD_TRY' columns.")

