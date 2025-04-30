
# 🔌 Energy Consumption Forecasting System

A modular, interactive system for forecasting energy usage using SARIMA, Prophet, XGBoost, and LSTM models. It provides both appliance-level and area-level predictions and supports dynamic cost estimation based on user inputs.

---

## 📂 Project Modules

### 🔹 Module 2: Appliance-Level Forecasting
- Dataset: [REFIT Household Energy Data](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)
- Models: SARIMA, Prophet
- Forecasts individual appliance usage (e.g., Fridge, Heater)
- Daily kWh prediction and graph visualization

### 🔹 Module 3: Group-Level Forecasting & Cost Estimation
- Dataset: [EnergyData_Complete (Kaggle)](https://www.kaggle.com/datasets/tuyntrnth/energydata-complete)
- Model: XGBoost
- Forecasts total usage with temperature & humidity as features
- Allows user input of appliances and usage duration for cost estimation

### 🔹 Module 1: Model Comparison Dashboard
- Dataset: EnergyData_Complete
- Models: ARIMA, XGBoost, LSTM
- Visual and metric-based comparison (MAE, RMSE, MAPE)
- Developed using Flask or Streamlit

---

## 🛠 Technologies Used

- **Python**, **Pandas**, **NumPy**
- **XGBoost**, **Statsmodels**, **Facebook Prophet**
- **TensorFlow / Keras** (for LSTM)
- **Flask / Streamlit** for frontend UI
- **Plotly** for graphs

---

## 📊 Features

- Forecast energy usage for selected appliances
- Display prediction graphs (hourly, daily)
- Cost calculator based on appliance wattage and hours of use
- Model comparison page (metrics + charts)
- Power-saving tips and visual suggestions

---

## 🚀 How to Run the Project

```bash
# Clone the repo
git clone https://github.com/ashrafsk359/Energy_Consumption_Forecasting_Time_series

# (Optional) Set up virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Run the web app
python app.py
```

---

## 📎 Notes

- Replace placeholders in code for model paths, dataset paths, or streamlit/flask selection.
- If datasets are not included, download from the original source above.

---

## 👨‍💻 Contributors

- **Ashraf Sulthan** – Module 2 & UI design
- **Srikanth** – XGBoost implementation and synthetic forecasting
- **Rajendra** – Comparative dashboard with ARIMA, LSTM

---

## 📜 License

This project is licensed under the MIT License.
