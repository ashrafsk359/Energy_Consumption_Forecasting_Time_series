import pandas as pd
from prophet import Prophet
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima  

# Define appliances for different models
prophet_appliances = ['Fridge', 'Chest Freezer', 'Upright Freezer', 'Computer Site', 'Television Site']
sarima_appliances = ['Tumble Dryer', 'Washing Machine', 'Dishwasher', 'Electric Heater']

def train_prophet(data_file):
    """ Train Prophet models for steady usage appliances """
    df = pd.read_csv(data_file)

    for appliance in prophet_appliances:
        print(f"🔍 Training Prophet model for {appliance}...")

        appliance_df = df[['ds', appliance, 'temperature', 'is_weekend']].rename(columns={appliance: 'y'}).dropna()

        # Prophet model with optimized seasonality & trend flexibility
        model = Prophet(
            changepoint_prior_scale=0.15,  # More flexibility for trend shifts
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True
        )

        # Add external regressors
        model.add_regressor('temperature')
        model.add_regressor('is_weekend')

        # Add monthly seasonality
        model.add_seasonality(name='monthly', period=30.5, fourier_order=10)

        model.fit(appliance_df)

        with open(f"../models/prophet_model_{appliance}.pkl", 'wb') as f:
            pickle.dump(model, f)

        print(f"✅ {appliance} Prophet model saved.")

def train_sarima(data_file):
    """ Train SARIMA models for appliances with irregular peaks """
    df = pd.read_csv(data_file)

    for appliance in sarima_appliances:
        print(f"🔍 Finding best SARIMA order for {appliance}...")

        appliance_df = df[['ds', appliance]].dropna()

        # Auto ARIMA to find best SARIMA parameters
        best_model = auto_arima(
            appliance_df[appliance],
            seasonal=True,
            m=7,  # Weekly seasonality
            trace=True,
            suppress_warnings=True
        )

        # Train SARIMA with best parameters found
        model = SARIMAX(appliance_df[appliance], order=best_model.order, seasonal_order=best_model.seasonal_order)
        model_fit = model.fit(disp=False)

        with open(f"../models/sarima_model_{appliance}.pkl", 'wb') as f:
            pickle.dump(model_fit, f)

        print(f"✅ {appliance} SARIMA model saved.")

if __name__ == "__main__":
    data_file = '../data/processed_data.csv'
    train_prophet(data_file)
    train_sarima(data_file)
