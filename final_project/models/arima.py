import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load training data
train = pd.read_csv("data/y_train.csv", index_col='date', parse_dates=True)
test = pd.read_csv("data/y_test.csv", index_col='date', parse_dates=True)

# Check stationarity using Augmented Dickey-Fuller (ADF) test
def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is NOT stationary, differencing needed.")

check_stationarity(train)

# Apply differencing if needed
train_diff = train.diff().dropna()  # First-order differencing
check_stationarity(train_diff)

# Plot ACF and PACF to determine AR (p) and MA (q)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sm.graphics.tsa.plot_acf(train_diff, lags=20, ax=ax[0])
sm.graphics.tsa.plot_pacf(train_diff, lags=20, ax=ax[1])
plt.show()

train.index.freq = '10min'
test.index.freq = '10min'

# Define ARIMA model (p, d, q) - Update these values based on ACF/PACF plots
p, d, q = 2, 1, 2  # Example values, adjust accordingly

# Train ARIMA model
arima_model = ARIMA(train, order=(p, d, q))
arima_result = arima_model.fit()

# Forecast on test set
arima_forecast = arima_result.forecast(steps=len(test))

# Convert test and forecast to numpy arrays
test_values = test.values.flatten()
arima_forecast_values = arima_forecast.values.flatten()

# Evaluate model performance
mae = mean_absolute_error(test_values, arima_forecast_values)
rmse = np.sqrt(mean_squared_error(test_values, arima_forecast_values))
mape = np.mean(np.abs((test_values - arima_forecast_values) / (test_values + 1e-10))) * 100

print(f"ARIMA Performance:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(test.index, test_values, label="Actual", color='blue')
plt.plot(test.index, arima_forecast_values, label="ARIMA Forecast", color='red')
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()
