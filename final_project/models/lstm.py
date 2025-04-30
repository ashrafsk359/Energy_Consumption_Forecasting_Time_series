import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load Data
train = pd.read_csv("data/y_train.csv", index_col='date', parse_dates=True)
test = pd.read_csv("data/y_test.csv", index_col='date', parse_dates=True)

# Normalize Data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Prepare Data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 10  # Using past 10 time steps
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train Model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Forecast
lstm_predictions = model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate Model
mae = mean_absolute_error(y_test_actual, lstm_predictions)
rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
mape = np.mean(np.abs((y_test_actual - lstm_predictions) / (y_test_actual + 1e-10))) * 100

print(f"LSTM Performance:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%")

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(test.index[sequence_length:], y_test_actual, label='Actual', color='blue')
plt.plot(test.index[sequence_length:], lstm_predictions, label='LSTM Forecast', color='red')
plt.legend()
plt.title("LSTM Forecast vs Actual")
plt.show()