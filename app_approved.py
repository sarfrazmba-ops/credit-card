import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('best_model_approved.pkl')

st.title('Credit Approval Prediction')

st.write('Enter the details to predict if the credit application will be approved:')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
income = st.number_input('Income', min_value=0.0, value=50000.0)
years_at_job = st.number_input('Years at Job', min_value=0, max_value=50, value=5)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
existing_credit_cards = st.number_input('Existing Credit Cards', min_value=0, max_value=10, value=1)

if st.button('Predict'):
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'years_at_job': [years_at_job],
        'credit_score': [credit_score],
        'existing_credit_cards': [existing_credit_cards]
    })
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    if prediction == 1:
        st.write(f'Prediction: Approved (Probability: {prob[1]:.2f})')
    else:
        st.write(f'Prediction: Not Approved (Probability: {prob[0]:.2f})')