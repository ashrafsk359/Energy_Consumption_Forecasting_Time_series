import pandas as pd
from prophet import Prophet
import pickle

selected_appliances = ['Fridge', 'Chest Freezer', 'Upright Freezer', 'Computer Site', 'Television Site']

def train_prophet_model(data_file):
    df = pd.read_csv(data_file)

    for appliance in selected_appliances:
        print(f"🔍 Training Prophet model for {appliance}...")

        appliance_df = df[['ds', appliance]].rename(columns={appliance: 'y'}).dropna()

        model = Prophet(weekly_seasonality=True, changepoint_prior_scale=0.05)
        model.fit(appliance_df)

        with open(f"../models/prophet_model_{appliance}.pkl", 'wb') as f:
            pickle.dump(model, f)

        print(f"✅ {appliance} model saved.")

if __name__ == "__main__":
    train_prophet_model('../data/processed_data.csv')
