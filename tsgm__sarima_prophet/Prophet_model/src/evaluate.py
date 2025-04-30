import pandas as pd
from sklearn.metrics import mean_absolute_error

appliances = ['Fridge', 'Chest Freezer', 'Upright Freezer', 'Tumble Dryer', 
              'Washing Machine', 'Dishwasher', 'Computer Site', 
              'Television Site', 'Electric Heater', 'y']

def evaluate_forecast(actual_file, forecast_file):
    actual = pd.read_csv(actual_file)
    forecast = pd.read_csv(forecast_file)

    df = pd.merge(actual, forecast, on='ds', how='inner')

    print("📊 Evaluating Predictions in kWh (Converted from Watts)...\n")

    for appliance in appliances:
        actual_col = appliance if appliance in df.columns else appliance + '_x'
        predicted_col = appliance if appliance in df.columns else appliance + '_y'

        if actual_col in df.columns and predicted_col in df.columns:
            df[actual_col] = df[actual_col] / 1000  # Convert to kWh
            df[predicted_col] = df[predicted_col] / 1000  # Convert to kWh

            mae = mean_absolute_error(df[actual_col], df[predicted_col])
            print(f"📊 {appliance} MAE (kWh): {mae:.4f}")

if __name__ == "__main__":
    evaluate_forecast('../data/processed_data.csv', '../data/appliance_forecast.csv')
