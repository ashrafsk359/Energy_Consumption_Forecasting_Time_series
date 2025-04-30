import pandas as pd
from prophet import Prophet
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Define appliances for different models
prophet_appliances = ['Fridge', 'Chest Freezer', 'Upright Freezer', 'Computer Site', 'Television Site']
sarima_appliances = ['Tumble Dryer', 'Washing Machine', 'Dishwasher', 'Electric Heater']

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def train_prophet(data_file):
    """ Train Prophet models for steady usage appliances """
    df = pd.read_csv(data_file)

    for appliance in prophet_appliances:
        print(f"🔍 Training Prophet model for {appliance}...")

        if appliance not in df.columns:
            print(f"⚠️ Skipping {appliance}: Column missing in dataset.")
            continue
        
        appliance_df = df[['ds', appliance]].rename(columns={appliance: 'y'}).dropna()

        if appliance_df.empty:
            print(f"⚠️ Skipping {appliance}: No valid data.")
            continue

        model = Prophet(changepoint_prior_scale=0.1, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(appliance_df)

        with open(f"models/prophet_model_{appliance}.pkl", 'wb') as f:
            pickle.dump(model, f)

        print(f"✅ {appliance} Prophet model saved.")

def train_sarima(data_file):
    """ Train SARIMA models for appliances with irregular peaks """
    df = pd.read_csv(data_file)

    for appliance in sarima_appliances:
        print(f"🔍 Training SARIMA model for {appliance}...")

        if appliance not in df.columns:
            print(f"⚠️ Skipping {appliance}: Column missing in dataset.")
            continue
        
        appliance_df = df[['ds', appliance]].dropna()

        if appliance_df.empty:
            print(f"⚠️ Skipping {appliance}: No valid data.")
            continue

        try:
            seasonal_order = (1, 1, 1, 30) if appliance in ['Washing Machine', 'Dishwasher'] else (1, 1, 1, 7)
            model = SARIMAX(appliance_df[appliance], order=(1, 1, 1), seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False, maxiter=200)  # Increased max iterations

            # Save model
            model_fit.save(f"models/sarima_model_{appliance}.pkl")
            
            print(f"✅ {appliance} SARIMA model saved.")
        
        except Exception as e:
            print(f"❌ Error training SARIMA for {appliance}: {e}")

if __name__ == "__main__":
    data_file = '../data/processed_data.csv'
    train_prophet(data_file)
    train_sarima(data_file)
