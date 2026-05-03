import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('best_model.pkl')

st.title('Credit Score Prediction')

st.write('Enter the details to predict the credit score:')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
income = st.number_input('Income', min_value=0.0, value=50000.0)
years_at_job = st.number_input('Years at Job', min_value=0, max_value=50, value=5)
existing_credit_cards = st.number_input('Existing Credit Cards', min_value=0, max_value=10, value=1)
approved = st.selectbox('Approved', [0, 1])

if st.button('Predict'):
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'years_at_job': [years_at_job],
        'existing_credit_cards': [existing_credit_cards],
        'approved': [approved]
    })
    prediction = model.predict(input_data)[0]
    st.write(f'Predicted Credit Score: {prediction:.2f}')