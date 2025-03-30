

#!pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit

import pandas as pd
import numpy as np


import pandas as pd
import streamlit as st



# Read the CSV into a DataFrame
df = pd.read_csv('vehicles.csv')



# Load data


# Drop irrelevant columns
cols_to_drop = ['id', 'url', 'region_url', 'image_url', 'description', 'county']
df = df.drop(columns=cols_to_drop, errors='ignore')

# Handle missing values
df = df.dropna(subset=['price', 'year', 'odometer', 'condition', 'manufacturer'])

# Remove outliers (e.g., cars priced < $500 or > $100k)
df = df[(df['price'] > 500) & (df['price'] < 100_000)]
df = df[df['year'] > 1990]  # Remove vintage cars (optional)

from sklearn.preprocessing import LabelEncoder

# Encode categorical features (e.g., manufacturer, condition)
label_encoders = {}
categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save encoders for deployment

from sklearn.preprocessing import MinMaxScaler

# Normalize mileage (odometer) and age
scaler = MinMaxScaler()
df['age'] = 2024 - df['year']  # Calculate car age
df[['age', 'odometer']] = scaler.fit_transform(df[['age', 'odometer']])

from sklearn.model_selection import train_test_split

X = df[['manufacturer', 'condition', 'odometer', 'age', 'cylinders']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Linear Regression (Baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"MAE (Linear Regression): {mean_absolute_error(y_test, y_pred_lr)}")
print(f"R² (Linear Regression): {r2_score(y_test, y_pred_lr)}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"MAE (Random Forest): {mean_absolute_error(y_test, y_pred_rf)}")

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"MAE (XGBoost): {mean_absolute_error(y_test, y_pred_xgb)}")

import joblib
joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

import streamlit as st
import joblib
import pandas as pd

# Load pre-trained model and encoders
model = joblib.load('xgb_model.pkl')  # Replace with your saved model
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Used Car Price Estimator 🚗")

# Input widgets
manufacturer = st.selectbox("Brand", options=label_encoders['manufacturer'].classes_)
condition = st.selectbox("Condition", options=label_encoders['condition'].classes_)
odometer = st.number_input("Mileage", min_value=0, max_value=500000)
year = st.number_input("Year", min_value=1990, max_value=2024)
cylinders = st.selectbox("Cylinders", options=label_encoders['cylinders'].classes_)

# Preprocess inputs
input_data = pd.DataFrame({
    'manufacturer': [manufacturer],
    'condition': [condition],
    'odometer': [odometer],
    'age': [2024 - year],
    'cylinders': [cylinders]
})

# Encode categorical features
for col in ['manufacturer', 'condition', 'cylinders']:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Normalize numerical features
input_data[['age', 'odometer']] = scaler.transform(input_data[['age', 'odometer']])

# Predict
if st.button("Estimate Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")

