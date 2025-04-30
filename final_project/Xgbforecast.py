import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import load

# Define paths
output_folder = "outputs"
visualization_folder = os.path.join(output_folder, "visualization")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(visualization_folder, exist_ok=True)

# Load the trained XGBoost model
model_path = "models/xgb_model.pkl"  # Ensure the model is saved in this path
model = load(model_path)

# Load the last available data point for forecasting
historical_data = pd.read_csv("data/preprocessed_power_consumption.csv")  # Ensure this file exists
last_known_values = historical_data.iloc[-1, -10:].values.reshape(1, -1)

# Define forecast period (3 months ~ 90 days)
n_days = 90
future_forecasts = []

# Generate forecasts
for _ in range(n_days):
    next_prediction = model.predict(last_known_values)[0]
    future_forecasts.append(next_prediction)
    last_known_values = np.roll(last_known_values, -1)
    last_known_values[0, -1] = next_prediction  # Update with new prediction

# Create a DataFrame for saving results
forecast_dates = pd.date_range(start=historical_data['date'].iloc[-1], periods=n_days + 1, freq='D')[1:]
forecast_df = pd.DataFrame({"date": forecast_dates, "predicted_power": future_forecasts})
forecast_df.to_csv(os.path.join(output_folder, "xgb_forecast.csv"), index=False)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['date'], forecast_df['predicted_power'], marker='o', linestyle='-', label='XGBoost Forecast')
plt.xlabel('Date')
plt.ylabel('Predicted Power Consumption')
plt.title('XGBoost Power Consumption Forecast for Next 3 Months')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.savefig(os.path.join(visualization_folder, "xgb_forecast.png"))
plt.show()

print("Forecast saved in outputs/xgb_forecast.csv and visualization saved in outputs/visualization/xgb_forecast.png")
