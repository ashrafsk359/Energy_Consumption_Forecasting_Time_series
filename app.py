import os
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, session
from flask_session import Session
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import random
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import subprocess
import webbrowser
import bcrypt

# Set Matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Configure Flask-Session
app.config['SECRET_KEY'] = 'srikanth-1926'  # Replace with a secure random key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Ensure 'data' and 'static' directories exist
data_folder = r"D:\final\TSGM_For_Power_Consumption\data"
static_folder = r"D:\final\TSGM_For_Power_Consumption\static"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(static_folder, exist_ok=True)

# Path to users CSV
users_csv_path = os.path.join(data_folder, "users.csv")

# Initialize users CSV if it doesn't exist
def init_users_csv():
    if not os.path.exists(users_csv_path):
        df = pd.DataFrame(columns=['username', 'email', 'password'])
        df.to_csv(users_csv_path, index=False)

init_users_csv()

# Helper functions for CSV operations
def read_users():
    try:
        return pd.read_csv(users_csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['username', 'email', 'password'])

def save_user(username, email, password):
    users_df = read_users()
    new_user = pd.DataFrame([[username, email, password]], columns=['username', 'email', 'password'])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(users_csv_path, index=False)

def find_user_by_email(email):
    users_df = read_users()
    user = users_df[users_df['email'] == email]
    return user if not user.empty else None

# Load dataset for synthetic data processing
df_path = os.path.join(data_folder, "power_consumption.csv")
df = pd.read_csv(df_path, parse_dates=['date'])
df.set_index('date', inplace=True)

# Select numeric columns for synthetic data
df_numeric = df.select_dtypes(include=[np.number])

# Train Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=5, random_state=42).fit(df_numeric)
synthetic_data = gmm.sample(len(df_numeric))[0]
synthetic_df_temp = pd.DataFrame(synthetic_data, columns=df_numeric.columns)

# Generate synthetic data for the future
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=365*26, freq='D')
synthetic_data = gmm.sample(len(future_dates))[0]
synthetic_df = pd.DataFrame(synthetic_data, columns=df_numeric.columns)
synthetic_df['date'] = future_dates
synthetic_df.set_index('date', inplace=True)
synthetic_df.index = synthetic_df.index.normalize()  # Normalize index for consistency

# Save synthetic data to CSV file
synthetic_data_path = os.path.join(data_folder, "synthetic_power_consumption.csv")
synthetic_df.to_csv(synthetic_data_path)

# Define new target features
target_features = ['lights', 'tv', 'fans', 'fridge', 'ac']

# Feature Engineering for target features
for feature in target_features:
    df[f'{feature}_lag1'] = df[feature].shift(1)
    df[f'{feature}_lag2'] = df[feature].shift(2)

df.dropna(inplace=True)

# Split features and targets
X = df.drop(columns=target_features)
y = df[target_features]  # Multi-output targets

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a separate model for each target feature
models = {}
for feature in target_features:
    models[feature] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    models[feature].fit(X_train, y_train[feature])

# Example prediction
predictions = {feature: models[feature].predict(X_test) for feature in target_features}

# Create static plots once to avoid redundant processing
def create_plots():
    combined_data = np.vstack([df_numeric, synthetic_df, synthetic_df[df_numeric.columns]])
    embedding = umap.UMAP(random_state=42).fit_transform(combined_data)

    plots = [
        ('power_consumption_over_time.png', df['Appliances'], 'Power Consumption Over Time', 'Date', 'Power Consumption (Wh)'),
        ('distribution_appliances.png', df['Appliances'], 'Distribution of Appliances Power Consumption', 'Power Consumption (Wh)', 'Frequency')
    ]
    
    for file, data, title, xlabel, ylabel in plots:
        plt.figure(figsize=(10, 6))
        if 'Distribution' in title:
            sns.histplot(data, bins=30, kde=True)
        else:
            plt.plot(df.index, data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(f'static/{file}')
        plt.close()

    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('static/correlation_matrix.png')
    plt.close()

    # UMAP Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:len(df_numeric), 0], embedding[:len(df_numeric), 1], c='green', label='Original', alpha=0.5)
    plt.scatter(embedding[len(df_numeric)+len(synthetic_df):, 0], embedding[len(df_numeric)+len(synthetic_df):, 1], c='yellow', label='Synthetic', alpha=0.5)
    plt.title('UMAP: Original vs Synthetic')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.savefig('static/umap_visualization.png')
    plt.close()

    # Actual vs Predicted for multiple targets
    for feature in target_features:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test[feature], label='Actual')
        plt.plot(y_test.index, predictions[feature], label='Predicted')
        plt.title(f'Actual vs Predicted {feature} Power Consumption')
        plt.xlabel('Date')
        plt.ylabel(f'{feature} Power Consumption (Wh)')
        plt.legend()
        plt.savefig(f'static/actual_vs_predicted_{feature}.png')
        plt.close()

create_plots()

# Login required decorator
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', error="Please log in to access this page"))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = find_user_by_email(email)
        if user is None:
            error = "Invalid email or password"
        else:
            stored_password = user['password'].iloc[0].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                session['user_id'] = user['username'].iloc[0]
                success = "Login successful!"
                return redirect(url_for('home'))
            else:
                error = "Invalid email or password"

    return render_template('login.html', error=error, success=success)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm-password')

        if password != confirm_password:
            error = "Passwords do not match"
        else:
            users_df = read_users()
            if username in users_df['username'].values or email in users_df['email'].values:
                error = "Username or email already exists"
            else:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                save_user(username, email, hashed_password.decode('utf-8'))
                success = "Registration successful! Please log in."
                return redirect(url_for('login', success=success))

    return render_template('register.html', error=error, success=success)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('welcome'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/run_streamlit1')
def run_streamlit1():
    subprocess.Popen(["streamlit", "run", "final_project/app.py", "--server.port=8501"])
    webbrowser.open("http://localhost:8501")
    return "Launching Streamlit App 1..."

@app.route('/run_streamlit2')
def run_streamlit2():
    subprocess.Popen(["streamlit", "run", "tsgm__sarima_prophet/Prophet_model/app/app.py", "--server.port=8502"])
    webbrowser.open("http://localhost:8502")
    return "Launching Streamlit App 2..."

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/umap')
def umap_visualization():
    return send_file('static/umap_visualization.png', mimetype='image/png')

@app.route('/historical_plots')
def historical_plots():
    return send_file('static/power_consumption_over_time.png', mimetype='image/png')

@app.route('/view_synthetic_data')
def view_synthetic_data():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_pages = (len(synthetic_df) // per_page) + 1
    start = (page - 1) * per_page
    end = start + per_page
    table_html = synthetic_df.iloc[start:end].to_html(classes='data-table', border=0)
    return render_template('synthetic_data.html', table_html=table_html, page=page, total_pages=total_pages)

@app.route('/future_prediction', methods=['GET', 'POST'])
def future_prediction():
    prediction = None
    if request.method == 'POST':
        try:
            user_date = pd.to_datetime(request.form['date']).normalize()
            if user_date not in synthetic_df.index:
                min_date = synthetic_df.index.min().date()
                max_date = synthetic_df.index.max().date()
                prediction = (
                    f"Selected date {user_date.date()} is not available in synthetic data.<br>"
                    f"Please select a date between {min_date} and {max_date}."
                )
            else:
                daily_predictions = {
                    feature: synthetic_df.loc[user_date, feature]
                    for feature in target_features
                }
                total_consumption = sum(daily_predictions.values())
                max_feature = max(daily_predictions, key=daily_predictions.get)
                max_consumption = daily_predictions[max_feature]
                prediction = (
                    f"Predicted consumption on {user_date.date()}:<br>"
                    f"Total Power Consumption: {total_consumption:.2f} Wh<br>"
                    f"Feature with the highest consumption: {max_feature} ({max_consumption:.2f} Wh)<br>"
                )
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('future_prediction.html', prediction=prediction)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/calculate_power_consumption')
def calculate_power_consumption():
    return render_template('calculate_power.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)