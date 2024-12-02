
import streamlit as st
import pandas as pd
import pickle

def output_page():
    st.title("Prediction Result")

    # Load the model and scaler
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Prepare input data
    input_data = pd.DataFrame([st.session_state.inputs])

    # Preprocess input_data if necessary
    # For example, encoding categorical variables, scaling, etc.

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader(f"The predicted Claim Injury Type is: **{prediction}**")

    # Save option
    if st.button("Save Result"):
        result_df = pd.DataFrame([{'Prediction': prediction, **st.session_state.inputs}])
        result_df.to_csv('prediction_result.csv', index=False)
        st.success("Result saved as prediction_result.csv")

    # Return to Welcome Page
    if st.button("Return to Welcome Page"):
        st.session_state.page = 'welcome'



