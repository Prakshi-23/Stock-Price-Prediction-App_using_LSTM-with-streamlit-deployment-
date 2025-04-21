# 📈 Stock Price Prediction Using LSTM

This project demonstrates how to use **Long Short-Term Memory (LSTM)**, a type of Recurrent Neural Network (RNN), to predict stock prices based on historical data. It includes steps for data preprocessing, sequence generation, LSTM model training, and performance visualization.

---

### 🛠️ Features

- Loads historical stock data
- Normalizes data using `MinMaxScaler`
- Prepares input sequences with configurable `time_step`
- Builds and trains an LSTM-based neural network
- Predicts future stock prices
- Visualizes actual vs. predicted prices

---

### 📂 Project Structure

```
Stock_Price_prediction_LSTM.ipynb
README.md
```

---

### 📦 Requirements

Install the required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

### 📊 How It Works

1. **Data Preprocessing**
   - Loads stock price data (e.g., 'Close' prices).
   - Normalizes using `MinMaxScaler` to scale values between 0 and 1.

2. **Sequence Generation**
   - Uses a `prepare_data()` function that creates input-output pairs based on a given `time_step` (default: 60).

3. **LSTM Model**
   - A sequential Keras model with stacked LSTM layers and Dense output.
   - Trained on historical sequences to predict future stock prices.

4. **Evaluation & Visualization**
   - Predicts on test data.
   - Visualizes actual vs. predicted prices using Matplotlib.

---

### ⚙️ Key Function: `prepare_data()`

```python
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(X), np.array(y), scaler
```

> This function generates sequences and targets from time series data, ready for LSTM input.

---

### 📉 Sample Output

- Graph comparing actual vs predicted prices
- Evaluation metrics (like RMSE, optional)

---
