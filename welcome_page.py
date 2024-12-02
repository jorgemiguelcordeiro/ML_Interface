

import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Feature names in the correct order
feature_names = ['BMI', 'Age', 'HighBP', 'HighChol', 'GenHlth', 'PhysHlth', 'MentHlth', 'Smoker', 'PhysActivity', 'Sex']

# Form to collect data
st.title("Diabetes Prediction")
data = []
for feature in feature_names:
    # Use an appropriate input widget based on data type
    data.append(st.number_input(f'Enter {feature}', format="%.2f"))

# Button to make prediction
if st.button('Predict'):
    try:
        input_data = np.array([data])
        input_data_scaled = scaler.transform(input_data)
        y_pred_prob = model.predict_proba(input_data_scaled)[:, 1]
        probability = y_pred_prob[0] * 100  # Convert to percentage
        
        # Multi-level threshold evaluation
        if probability >= 75:
            result = "The patient has a high likelihood of having diabetes."
        elif 50 <= probability < 75:
            result = "The patient has a moderate likelihood of having diabetes."
        elif 30 <= probability < 50:
            result = "The patient has a low likelihood of having diabetes."
        else:
            result = "The patient is unlikely to have diabetes."
        
        st.write(f'Result: {result}')
        st.write(f'Probability of Diabetes: {probability:.2f}%')
    except Exception as e:
        st.error(f"Error: {str(e)}")


