import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load data
df = pd.read_csv('credit (3) (1).csv')

# Display data info
print(df.head())
print(df.info())
print(df.describe())

# Assume target is 'credit_score' for regression
target = 'credit_score'
features = ['age', 'income', 'years_at_job', 'existing_credit_cards', 'approved']

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{name}: R2 = {score:.4f}, MSE = {mse:.4f}')
    if score > best_score:
        best_score = score
        best_model = model

print(f'Best model: {best_model.__class__.__name__} with R2 = {best_score:.4f}')

# Save best model
import joblib
joblib.dump(best_model, 'best_model.pkl')