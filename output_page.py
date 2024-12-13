import streamlit as st
import pandas as pd
import pickle


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

    # Check if session state contains inputs
    if 'inputs' not in st.session_state:
        st.error("No input data found in session state. Please return to the input page.")
        return

    # Debug: Inspect session state
    st.write("Session State at Output Page:", st.session_state.inputs)

    # Prepare input data
    inputs = pd.DataFrame([st.session_state.inputs])  # Convert session state inputs to a DataFrame

    # Debug: Inspect raw input data
    st.write("Raw Input Data Before Processing:", inputs)

    # Drop unused columns
    unused_columns = ['OIICS Nature of Injury Description']
    inputs = inputs.drop(columns=unused_columns, errors='ignore')
           

    # Debug: Inspect input data after adding missing columns
    st.write("Input Data After Adding Missing Columns:", inputs)

    # Apply mappings to convert categorical variables into numerical representations
    try:
        if 'Gender' in inputs.columns:
            inputs['Gender'] = inputs['Gender'].map({'Female': 0, 'Male': 1}).fillna(0)
        if 'Alternative Dispute Resolution' in inputs.columns:
            inputs['Alternative Dispute Resolution'] = inputs['Alternative Dispute Resolution'].map({'Yes': 1, 'No': 0}).fillna(0)
        if 'Attorney/Representative' in inputs.columns:
            inputs['Attorney/Representative'] = inputs['Attorney/Representative'].map({'No': 0, 'Yes': 1}).fillna(0)
        if 'COVID-19 Indicator' in inputs.columns:
            inputs['COVID-19 Indicator'] = inputs['COVID-19 Indicator'].map({'No': 0, 'Yes': 1}).fillna(0)
    except Exception as e:
        st.error(f"An error occurred during mapping: {e}")
        return

    # Debug: Inspect processed input data
    st.write("Processed Input Data:", inputs)

    # Perform prediction
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

