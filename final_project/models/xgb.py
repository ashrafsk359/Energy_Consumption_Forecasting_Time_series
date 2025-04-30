import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
train = pd.read_csv("data/y_train.csv", index_col='date', parse_dates=True)
test = pd.read_csv("data/y_test.csv", index_col='date', parse_dates=True)

# Convert time series to supervised learning format
def create_features(data, lag=10):
    df = data.copy()
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df.iloc[:, 0].shift(i)  # Select only the first column before shifting
    df.dropna(inplace=True)
    return df

# Apply feature transformation
lag = 10  # Using 10 previous timesteps as features
train_supervised = create_features(train, lag)
test_supervised = create_features(test, lag)

# Split into features (X) and target (y)
X_train, y_train = train_supervised.drop(columns=['Appliances']), train_supervised['Appliances']
X_test, y_test = test_supervised.drop(columns=['Appliances']), test_supervised['Appliances']

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_forecast = xgb_model.predict(X_test)

# Evaluate Performance
mae = mean_absolute_error(y_test, xgb_forecast)
rmse = np.sqrt(mean_squared_error(y_test, xgb_forecast))
mape = np.mean(np.abs((y_test - xgb_forecast) / (y_test + 1e-10))) * 100

print(f"XGBoost Performance:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%")

# Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual", color='blue')
plt.plot(y_test.index, xgb_forecast, label="XGBoost Forecast", color='green')
plt.legend()
plt.title("XGBoost Forecast vs Actual")
plt.show()
from joblib import dump
import os

# Define path to save the model
model_path = "models/xgb_model.pkl"
os.makedirs("models", exist_ok=True)  # Ensure the folder exists

# Save the trained XGBoost model
dump(xgb_model, model_path)
print(f"Model saved at {model_path}")
