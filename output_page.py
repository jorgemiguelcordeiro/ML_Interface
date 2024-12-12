import streamlit as st
import pandas as pd
import pickle
import os

def output_page():
    st.title("Prediction Result")

    # Get the absolute path to the directory containing this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the model file
    model_path = os.path.join(BASE_DIR, 'model.pkl')

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
    input_data = pd.DataFrame([st.session_state.inputs])

    # Drop columns that are not used in the model
    unused_columns = ['OIICS Nature of Injury Description']
    input_data = input_data.drop(columns=unused_columns, errors='ignore')

    # Preprocess input_data if necessary
    # For example, encoding categorical variables, scaling, etc.

    # Handle exceptions during prediction
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
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


