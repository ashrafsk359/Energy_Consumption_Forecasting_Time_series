import pandas as pd
import os  # To check if temperature file exists

def preprocess_energy_data(input_file, temp_file, output_file):
    df = pd.read_csv(input_file, parse_dates=['Time'])
    df['Date'] = df['Time'].dt.date  

    # Aggregate consumption to daily level
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    daily_df = df.groupby('Date')[numeric_cols].sum().reset_index()

    # Add weekend effect (1 for Sat/Sun, 0 for weekdays)
    daily_df['ds'] = pd.to_datetime(daily_df['Date'])
    daily_df['weekday'] = daily_df['ds'].dt.weekday
    daily_df['is_weekend'] = daily_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Check if temperature file exists before merging
    if os.path.exists(temp_file):
        temp_df = pd.read_csv(temp_file)
        temp_df['ds'] = pd.to_datetime(temp_df['ds'])  
        daily_df = pd.merge(daily_df, temp_df[['ds', 'temperature']], on='ds', how='left')
    else:
        print("⚠️ Temperature data not found. Skipping temperature feature.")
        daily_df['temperature'] = 0  # Default to 0 if not available

    # Rename for Prophet compatibility
    daily_df.rename(columns={'Aggregate': 'y'}, inplace=True)
    daily_df.to_csv(output_file, index=False)
    print(f"✅ Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_energy_data('../data/House1_hourly.csv', '../data/temperature.csv', '../data/processed_data.csv')
