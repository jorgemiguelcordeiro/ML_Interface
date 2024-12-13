import streamlit as st
import pandas as pd
import pickle
import os
def output_page():
    st.title("Prediction Result")

    # Load the model
    model_path = 'logistic_model.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.write("Model loaded successfully.")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'logistic_model.pkl' is in the correct directory.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return

    # Prepare input data
    inputs = pd.DataFrame([st.session_state.inputs])

    # Drop unused columns
    unused_columns = ['OIICS Nature of Injury Description']
    inputs = inputs.drop(columns=unused_columns, errors='ignore')

    # Add missing columns
    required_columns = ['Gender', 'Alternative Dispute Resolution', 'Attorney/Representative', 'COVID-19 Indicator']
    for col in required_columns:
        if col not in inputs.columns:
            st.warning(f"'{col}' is missing. Adding it with default values.")
            inputs[col] = 0

    # Apply mappings
    try:
        inputs['Gender'] = inputs['Gender'].map({'Female': 0, 'Male': 1}).fillna(0)
        inputs['Alternative Dispute Resolution'] = inputs['Alternative Dispute Resolution'].map({'Yes': 1, 'No': 0}).fillna(0)
        inputs['Attorney/Representative'] = inputs['Attorney/Representative'].map({'No': 0, 'Yes': 1}).fillna(0)
        inputs['COVID-19 Indicator'] = inputs['COVID-19 Indicator'].map({'No': 0, 'Yes': 1}).fillna(0)
    except Exception as e:
        st.error(f"An error occurred during mapping: {e}")
        return

    st.write("Processed Input Data:", inputs)

    # Prediction
    try:
        prediction = model.predict(inputs)[0]
        st.subheader(f"The predicted Claim Injury Type is: **{prediction}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return

    # Save Results
    if st.button("Save Result"):
        result_df = pd.DataFrame([{'Prediction': prediction, **st.session_state.inputs}])
        result_df.to_csv('prediction_result.csv', index=False)
        st.success("Result saved as 'prediction_result.csv'")

    # Navigation
    if st.button("Return to Welcome Page"):
        st.session_state.page = 'welcome'



