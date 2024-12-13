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

    # Convert session state inputs to DataFrame
    inputs = pd.DataFrame([st.session_state.inputs])  
    
    # Debug: Show raw input data
    st.write("**Raw Input Data Before Processing:**", inputs)

    # Check required columns
    required_columns = ['gender', 'alternative_dispute_resolution', 'attorney_representative', 'covid_19_indicator']
    missing_required_cols = [col for col in required_columns if col not in inputs.columns]

    if missing_required_cols:
        for col in missing_required_cols:
            st.warning(f"'{col}' column is missing from the inputs. Please ensure it is captured in the input page.")
        # We do not return here; let's continue so we can inspect what's present

    # Debug: Show columns before dropping unused columns
    st.write("**Columns before dropping unused columns:**", inputs.columns.tolist())

    # Drop unused columns if present
    unused_columns = ['oiics_nature_of_injury_description']
    inputs = inputs.drop(columns=unused_columns, errors='ignore')

    # Debug: Show columns after dropping unused columns
    st.write("**Columns after dropping unused columns:**", inputs.columns.tolist())

    # Apply mappings if columns exist
    # Print unique values before mapping for debugging
    categorical_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'alternative_dispute_resolution': {'Yes': 1, 'No': 0},
        'attorney_representative': {'No': 0, 'Yes': 1},
        'covid_19_indicator': {'No': 0, 'Yes': 1}
    }

    for col, mapping in categorical_mappings.items():
        if col in inputs.columns:
            st.write(f"**Unique values in '{col}' before mapping:**", inputs[col].unique().tolist())
            inputs[col] = inputs[col].map(mapping)
            st.write(f"**Unique values in '{col}' after mapping:**", inputs[col].unique().tolist())
        else:
            st.warning(f"Skipping mapping for '{col}' because it does not exist in the DataFrame.")

    # Debug: Inspect processed input data
    st.write("**Processed Input Data:**", inputs)

    # Perform prediction if all needed columns are present
    try:
        # Attempt to predict only if we have the columns needed by the model
        # The model might require a certain set of columns. Ensure you know them.
        # Assuming the model requires the columns currently in 'inputs'
        prediction = model.predict(inputs)[0]
        st.subheader(f"The predicted Claim Injury Type is: **{prediction}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        # Don't return here, so we can still potentially save or debug further

    # Save Results
    if st.button("Save Result"):
        result_df = pd.DataFrame([{'Prediction': prediction, **st.session_state.inputs}])
        result_df.to_csv('prediction_result.csv', index=False)
        st.success("Result saved as 'prediction_result.csv'")

    # Navigation
    if st.button("Return to Welcome Page"):
        st.session_state.page = 'welcome'

