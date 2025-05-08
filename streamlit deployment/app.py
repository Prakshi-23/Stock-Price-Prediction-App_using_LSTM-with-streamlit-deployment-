import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st

tf.random.set_seed(42)
np.random.seed(42)

# Streamlit UI
st.title("Stock Price Prediction using LSTM")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, ^NSEI):", "^NSEI")

# Fetch Stock Data
def fetch_stock_data(ticker):
    stock = yf.download(ticker, period='max')
    return stock['Close']

# Prepare dataset for LSTM
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i - time_step:i, 0])
        y.append(data_scaled[i, 0])
    
    return np.array(X), np.array(y), scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict Future Prices
def predict_future(model, data, time_step, scaler, days=7):
    last_days = data.reshape(1, -1, 1)
    predictions = []
    
    for _ in range(days):
        pred = model.predict(last_days, verbose=0)
        predictions.append(pred[0, 0])
        last_days = np.append(last_days[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Run when user enters a ticker
if st.button("Predict"):
    with st.spinner("Fetching Data & Training Model..."):
        df = fetch_stock_data(ticker)
        X, y, scaler = prepare_data(df)
        
        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Train Model
        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=512, verbose=0)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # First Figure: Actual vs Predicted Prices
        fig1 = plt.subplots(figsize=(12, 6))[1]  
        fig1.plot(df.index[-len(y_test):], y_test_rescaled, label="Actual Price", color="blue")
        fig1.plot(df.index[-len(y_test):], y_pred_rescaled, label="Predicted Price", color="red")
        fig1.set_xlabel('Date')
        fig1.set_ylabel('Stock Price')
        fig1.set_title(f"{ticker} Stock Price Prediction")
        fig1.legend()
        st.pyplot(fig1.figure)  

        # Second Figure: Future Forecast
        future_preds = predict_future(model, X[-1], 60, scaler, days=30)
        future_dates = pd.date_range(start=df.index[-1], periods=30, freq='B')  # Business days only

        fig2 = plt.subplots(figsize=(12, 6))[1]  # Extract ax
        fig2.plot(df.index[-len(y_test):], y_test_rescaled, label="Actual Price", color="blue")
        fig2.plot(df.index[-len(y_test):], y_pred_rescaled, label="Predicted Price", color="red")
        fig2.plot(future_dates, future_preds, label="Forecasted Price", color="green", linestyle='dashed')
        fig2.set_xlabel('Date')
        fig2.set_ylabel('Stock Price')
        fig2.set_title(f"{ticker} Future Stock Price Prediction")
        fig2.legend()
        st.pyplot(fig2.figure)
        
    