import streamlit as st
import pandas as pd
import plotly.express as px
import os
import subprocess

# Set correct paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORECAST_SCRIPT = os.path.join(BASE_DIR, "src", "forecast.py")
FORECAST_FILE = os.path.join(BASE_DIR, "data", "appliance_forecast.csv")
ACTUAL_CONSUMPTION_FILE = os.path.join(BASE_DIR, "data", "House1_hourly.csv")  # Actual past consumption data

@st.cache_data
def load_forecast_data():
    """Load the appliance forecast data."""
    if not os.path.exists(FORECAST_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(FORECAST_FILE)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    
    if 'ds' not in df.columns:
        return pd.DataFrame()
    
    df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime format
    return df

@st.cache_data
def load_actual_data():
    """Load actual past consumption data from House1_hourly.csv."""
    if not os.path.exists(ACTUAL_CONSUMPTION_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(ACTUAL_CONSUMPTION_FILE)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    if 'Time' not in df.columns:
        return pd.DataFrame()
    
    df.rename(columns={'Time': 'ds'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime format
    return df

# Forecast generation function
def generate_forecast(date):
    """Run forecast.py dynamically if needed."""
    VENV_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    subprocess.run([VENV_PYTHON, FORECAST_SCRIPT, str(date)], check=True)

# Constants
APPLIANCES = ['Fridge', 'Chest Freezer', 'Upright Freezer', 'Tumble Dryer',
              'Washing Machine', 'Dishwasher', 'Computer Site', 
              'Television Site', 'Electric Heater']
ELECTRICITY_RATE = 10  # Cost per kWh in Rupees (₹)

# Streamlit UI
st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")
st.title("🔌 Energy Consumption Forecasting")

# Sidebar Inputs
st.sidebar.header("📅 Select Forecast Date")
forecast_date = st.sidebar.date_input("Choose a date", pd.to_datetime("2015-07-11"),
                                      min_value=pd.to_datetime("2015-07-11"), max_value=pd.to_datetime("2015-08-10"))
forecast_date = pd.to_datetime(forecast_date)

st.sidebar.header("⚡ Select Appliances")
selected_appliances = st.sidebar.multiselect("Choose appliances", APPLIANCES, default=APPLIANCES)

# Button to trigger forecast
if st.sidebar.button("🔮 Forecast"):
    df = load_forecast_data()
    actual_df = load_actual_data()

    # If forecast doesn't exist, generate it dynamically
    if df.empty or forecast_date.date() not in df["ds"].dt.date.unique():
        st.warning("⚠️ Generating forecast for the selected date... Please wait.")
        generate_forecast(forecast_date.date())
        df = load_forecast_data()  # Reload after generation

    if df.empty:
        st.error("⚠️ Forecast generation failed. Check forecast.py logs.")
    else:
        # Filter Data for Selected Date
        filtered_df = df[df['ds'].dt.date == forecast_date.date()].copy()
        if not filtered_df.empty:
            filtered_df["hour"] = filtered_df["ds"].dt.hour  # Ensure 'hour' column exists

        # Get Last 10 Days Actual Data
        past_10_days_df = actual_df[(actual_df['ds'].dt.date < forecast_date.date()) & 
                                    (actual_df['ds'].dt.date >= (forecast_date - pd.DateOffset(days=10)).date())].copy()
        if not past_10_days_df.empty:
            past_10_days_df["hour"] = past_10_days_df["ds"].dt.hour  # Ensure 'hour' column exists
            past_10_days_df["Total Consumption"] = past_10_days_df["Aggregate"] / 1000  # Convert Watts to kWh

        # Filter Data for the Entire Forecast Month
        monthly_df = df[(df['ds'] >= forecast_date) & (df['ds'] < forecast_date + pd.DateOffset(months=1))].copy()
        if not monthly_df.empty:
            monthly_df["day"] = monthly_df["ds"].dt.normalize().dt.date  # Ensure only date (no time)

        if filtered_df.empty:
            st.warning("⚠️ No forecast data available for the selected date.")
        else:
            # Convert Watts to kWh
            filtered_df["Total Consumption"] = filtered_df[selected_appliances].sum(axis=1) / 1000  
            monthly_df["Total Consumption"] = monthly_df[selected_appliances].sum(axis=1) / 1000  

            # 📊 **Today's Forecast vs. Last 10 Days Comparison**
            st.subheader("📅 Today's Forecast vs. Last 10 Days")

            if not past_10_days_df.empty:
                fig_daily = px.line(
                    past_10_days_df,
                    x="hour",
                    y="Total Consumption",
                    color=past_10_days_df["ds"].dt.date.astype(str),
                    title="Comparison of Today's Forecast with the Last 10 Days",
                    labels={"hour": "Time (Hours)", "Total Consumption": "Energy (kWh)"},
                    markers=True
                )
                fig_daily.add_scatter(
                    x=filtered_df["hour"],
                    y=filtered_df["Total Consumption"],
                    mode="lines+markers",
                    name="Predicted",
                    line=dict(dash="dash", color="red")
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.warning("⚠️ No past data available for comparison.")

            # **Daily Cost Estimation**
            daily_energy = filtered_df["Total Consumption"].sum()
            daily_cost = daily_energy * ELECTRICITY_RATE

            st.write(f"**🔹 Today's Energy Consumption:** {daily_energy:.2f} kWh")
            st.write(f"**💰 Estimated Cost for Today:** ₹{daily_cost:.2f} (at ₹{ELECTRICITY_RATE}/kWh)")

            # 📊 **Monthly Forecast**
            if not monthly_df.empty:
                st.subheader("📆 Monthly Forecast (Daily Predictions)")

                # Generate Daily Forecast for the Entire Month
                daily_consumption = monthly_df.groupby("day")["Total Consumption"].sum().reset_index()

                fig_monthly = px.line(
                    daily_consumption,
                    x="day",
                    y="Total Consumption",
                    title="Predicted Energy Consumption for the Month",
                    labels={"day": "Date", "Total Consumption": "Energy (kWh)"},
                    markers=True
                )
                fig_monthly.update_layout(
                    xaxis=dict(
                        tickformat="%b %d",
                        title="Date"
                    ),
                    yaxis_title="Energy (kWh)"
                )
                st.plotly_chart(fig_monthly, use_container_width=True)

                # **Monthly Cost Estimation**
                monthly_energy = monthly_df["Total Consumption"].sum()
                monthly_cost = monthly_energy * ELECTRICITY_RATE

                st.write(f"**🔹 Estimated Monthly Energy Consumption:** {monthly_energy:.2f} kWh")
                st.write(f"**💰 Estimated Monthly Cost:** ₹{monthly_cost:.2f} (at ₹{ELECTRICITY_RATE}/kWh)")

st.write("🔹 Select your options and click **'Forecast'** to generate predictions.")
