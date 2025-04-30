import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
file_path = "data/preprocessed_power_consumption.csv"
df = pd.read_csv(file_path, index_col='date', parse_dates=True)

# Define features (X) and target variable (y)
X = df.drop(columns=['Appliances'])  # Features
y = df['Appliances']  # Target variable

# Split the dataset into train (80%) and test (20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Save the train and test sets
X_train.to_csv("data/X_train.csv")
X_test.to_csv("data/X_test.csv")
y_train.to_csv("data/y_train.csv")
y_test.to_csv("data/y_test.csv")

print("Train-test split completed.")
print(f"Train Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")
