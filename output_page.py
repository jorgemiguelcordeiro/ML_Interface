import streamlit as st
import pandas as pd
import pickle
import os

def output_page():
    st.title("Prediction Result")

    # Construct the full path to the model file
    model_path = 'logistic_model.pkl'

    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
       
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the correct directory.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return
    


    # Prepare input data
    inputs = pd.DataFrame([st.session_state.inputs])

    # Drop columns that are not used in the model
    unused_columns = ['OIICS Nature of Injury Description']
    inputs = inputs.drop(columns=unused_columns, errors='ignore')

    # Check for available columns
    st.write("Input Data Columns:", inputs.columns)

    # Apply mapping for 'Alternative Dispute Resolution' if it exists
    if 'Alternative Dispute Resolution' in inputs.columns:
        inputs['Alternative Dispute Resolution'] = inputs['Alternative Dispute Resolution'].map({'Yes': 1, 'No': 0}).fillna(0)
    else:
        st.warning("'Alternative Dispute Resolution' column is missing in the input data.")

    # Apply mapping for 'Gender' if it exists
    if 'Gender' in input_data.columns:
        inputs['Gender'] = inputs['Gender'].map({'F': 0, 'M': 1}).fillna(0)
    else:
        st.warning("'Gender' column is missing in the input data.")

    # Apply mapping for 'Attorney/Representative' if it exists
    if 'Attorney/Representative' in input_data.columns:
        inputs['Attorney/Representative'] = inputs['Attorney/Representative'].map({'No': 0, 'Yes': 1}).fillna(0)
    else:
        st.warning("'Attorney/Representative' column is missing in the input data.")
    # Apply mapping for 'COVID-19 Indicator' if it exists
    if 'COVID-19 Indicator' in input_data.columns:
        inputs['COVID-19 Indicator'] = inputs['COVID-19 Indicator'].map({'No': 0, 'Yes': 1}).fillna(0)
    else:
        st.warning("'COVID-19 Indicator' column is missing in the input data.")

    st.write("Processed Input Data:", inputs)

    # Preprocess input_data if necessary
    # For example, encoding categorical variables, scaling, etc.

    # Handle exceptions during prediction
    try:
        # Make prediction
        prediction = model.predict(inputs)[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return

    st.subheader(f"The predicted Claim Injury Type is: **{prediction}**")

    # Save option
    if st.button("Save Result"):
        result_df = pd.DataFrame([{'Prediction': prediction, **st.session_state.inputs}])
        result_df.to_csv('prediction_result.csv', index=False)
        st.success("Result saved as 'prediction_result.csv'")

    # Return to Welcome Page
    if st.button("Return to Welcome Page"):
        st.session_state.page = 'welcome'


