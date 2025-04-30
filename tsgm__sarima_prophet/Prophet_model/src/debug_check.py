import pandas as pd
from prophet import Prophet

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Show available columns for debugging
print("\n📊 Available Columns in Data:", df.columns.tolist())

# Ensure correct datetime format
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")  # Fix datetime issues
df.dropna(subset=["ds"], inplace=True)  # Remove invalid dates

# Check dataset date range
print("\n📅 Available Data Range:", df["ds"].min(), "to", df["ds"].max())

# Find the correct column for total energy consumption
if "y" in df.columns:
    energy_col = "y"  # Assuming 'y' is total consumption
    print(f"\n✅ Using column '{energy_col}' for forecasting.")
else:
    print("\n❌ ERROR: No total energy consumption column found!")
    exit()

# Check if past 10 days' data exist (adjusting based on dataset range)
latest_date = df["ds"].max()
past_10_days = df[(df["ds"] >= latest_date - pd.Timedelta(days=10)) & (df["ds"] <= latest_date)]

print("\n✅ Checking if past 10 days exist:")
if past_10_days.empty:
    print("\n❌ ERROR: No past data available for comparison!")
else:
    print(past_10_days[["ds", energy_col]].tail(10))  # Show last 10 days' data

# Train Prophet model to check forecast behavior
model = Prophet(
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Select required columns
df = df[["ds", energy_col]].dropna()

try:
    model.fit(df)
    future = model.make_future_dataframe(periods=30)  # Forecast next 30 days
    forecast = model.predict(future)

    print("\n📊 Forecast Sample:")
    print(forecast[["ds", "yhat"]].head(10))  # Check predicted values

    # Check if trend is unexpectedly linear
    if forecast["yhat"].diff().mean() > 0:
        print("\n⚠️ WARNING: Forecast trend seems too linear. Check model training data!")
    else:
        print("\n✅ Forecast shows expected variations.")

except Exception as e:
    print("\n❌ ERROR in model training:", e)
