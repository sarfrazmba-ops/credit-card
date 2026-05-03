import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv('credit (3) (1).csv')

# Display data info
print(df.head())
print(df.info())
print(df.describe())

# Target is 'approved' for classification
target = 'approved'
features = ['age', 'income', 'years_at_job', 'credit_score', 'existing_credit_cards']

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f'{name}: Accuracy = {score:.4f}')
    print(classification_report(y_test, y_pred))
    if score > best_score:
        best_score = score
        best_model = model

print(f'Best model: {best_model.__class__.__name__} with Accuracy = {best_score:.4f}')

# Save best model
joblib.dump(best_model, 'best_model_approved.pkl')