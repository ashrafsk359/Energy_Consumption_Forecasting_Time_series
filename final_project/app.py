import streamlit as st
import pandas as pd

# Load your CSV file
data = pd.read_csv(r"D:\final\TSGM_For_Power_Consumption\final_project\outputs\xgb_forecast.csv")
  # Replace with your file path

# Title of the interface
st.title("Forecasting Dashboard")

# Sidebar with buttons
option = st.sidebar.selectbox(
    "Choose an Option",
    ("Forecasted Data", "Model Graphs", "Model Compare", "Forecasted by XGBoost", "Calculate Power Cost")
)

# Button actions
if option == "Forecasted Data":
    st.subheader("Forecasted Data")
    st.write(data)  # Display the CSV data as a table

elif option == "Model Graphs":
    st.subheader("Model Graphs")
    # Display three pre-generated graphs from outputs/visualizations
    st.image(r"D:/final/TSGM_For_Power_Consumption/final_project/outputs/visualizations/arima.png", 
             caption="ARIMA Model Graph", use_container_width=True)
    st.image(r"D:/final/TSGM_For_Power_Consumption/final_project/outputs/visualizations/lstm.png", 
             caption="LSTM Model Graph", use_container_width=True)
    st.image(r"D:/final/TSGM_For_Power_Consumption/final_project/outputs/visualizations/xgb.png", 
             caption="XGBoost Model Graph", use_container_width=True)

elif option == "Model Compare":
    st.subheader("Model Compare")
    # Display pre-generated graph
    st.image(r"D:/final/TSGM_For_Power_Consumption/final_project/outputs/model_comparison.png", 
             caption="Model Comparison", use_container_width=True)
    # Replace with your actual filename if different

elif option == "Forecasted by XGBoost":
    st.subheader("Forecasted by XGBoost")
    # Display pre-generated graph
    st.image(r"D:/final/TSGM_For_Power_Consumption/final_project/outputs/visualization/xgb_forecast.png", 
             caption="XGBoost Forecast", use_container_width=True)
    # Replace with your actual filename if different

elif option == "Calculate Power Cost":
    st.subheader("Calculate Power Cost")
    
    # Create a form for interactive input
    with st.form(key="power_cost_form"):
        st.write("### Energy Cost Calculator")
        rate_per_unit = st.number_input("Rate per unit (₹)", min_value=0.0, value=5.0, step=0.01, format="%.2f")
        
        # Calculate default daily usage (summing first 24 rows for one day)
        try:
            daily_usage = data["predicted_power"].iloc[:24].sum()  # Sum first day's worth (24 hours)
        except KeyError:
            daily_usage = 0.0  # Fallback if column name is wrong
            st.warning("No 'predicted_power' column found in CSV. Please enter daily usage manually.")
        
        power_usage = st.number_input("Daily Power Usage (kWh)", min_value=0.0, value=daily_usage, step=1.0, format="%.2f")
        submit_button = st.form_submit_button(label="Calculate Cost")

    # Display monthly cost (30 days) after button is clicked
    if submit_button:
        monthly_usage = power_usage * 30  # Monthly usage for 30 days
        monthly_cost = monthly_usage * rate_per_unit  # Cost for 30 days
        st.success(f"*Power Cost per Month: ₹{monthly_cost:.2f}*")
        st.write(f"Based on {power_usage:.2f} kWh per day, totaling {monthly_usage:.2f} kWh for 30 days at ₹{rate_per_unit:.2f} per unit.")