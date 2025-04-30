import pandas as pd
import pickle
import os

# Define appliance groups
prophet_appliances = ['Fridge', 'Chest Freezer', 'Upright Freezer', 'Computer Site', 'Television Site']
sarima_appliances = ['Tumble Dryer', 'Washing Machine', 'Dishwasher', 'Electric Heater']

def forecast_all(days=30):
    df = pd.DataFrame()
    df['ds'] = pd.date_range(start='2015-07-11', periods=days, freq='D')

    # Prophet Forecasting
    for appliance in prophet_appliances:
        model_path = f"models/prophet_model_{appliance}.pkl"

        if not os.path.exists(model_path):
            print(f"⚠️ Prophet model for {appliance} not found!")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        forecast = model.predict(df[['ds']])
        df[appliance] = forecast['yhat']  # Extract predicted values

        print(f"✅ {appliance} Prophet forecast generated.")

    # SARIMA Forecasting
    for appliance in sarima_appliances:
        model_path = f"models/sarima_model_{appliance}.pkl"

        if not os.path.exists(model_path):
            print(f"⚠️ SARIMA model for {appliance} not found!")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        future_values = model.forecast(steps=days)
        df[appliance] = future_values.values

        print(f"✅ {appliance} SARIMA forecast generated.")

    print("🔍 Final Forecast Data:\n", df.head())  # Debugging Output

    df.to_csv('data/appliance_forecast.csv', index=False)
    print("✅ Forecast saved successfully.")

if __name__ == "__main__":
    forecast_all()
