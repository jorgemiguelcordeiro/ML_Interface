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

    # Prepare input data
    inputs = pd.DataFrame([st.session_state.inputs])  # Convert session state inputs to a DataFrame

    # Debug: Inspect raw input data
    st.write("Raw Input Data Before Processing:", inputs)
    
   # Debug: Check why required columns might be missing
    required_columns = ['Gender', 'Alternative Dispute Resolution', 'Attorney/Representative', 'COVID-19 Indicator']
    for col in required_columns:
        if col not in inputs.columns:
            st.warning(f"'{col}' column is missing from the inputs. Adding it with default value 0.")
        # Debugging missing column
            if col not in st.session_state.inputs:
                st.error(f"'{col}' is not present in st.session_state.inputs. Check the input page to ensure this field is captured correctly.")
            else:
                st.error(f"'{col}' exists in st.session_state.inputs but is not included in the DataFrame. Possible data conversion issue.")
        

    # Drop unused columns
    unused_columns = ['OIICS Nature of Injury Description']
    inputs = inputs.drop(columns=unused_columns, errors='ignore')

    # Apply mappings to convert categorical variables into numerical representations
    mapping_errors = {}
    try:
        if 'Gender' in inputs.columns:
            inputs['Gender'] = inputs['Gender'].map({'Female': 0, 'Male': 1})
            unmapped_gender = inputs['Gender'].isnull().sum()
            if unmapped_gender > 0:
                mapping_errors['Gender'] = unmapped_gender

        if 'Alternative Dispute Resolution' in inputs.columns:
            inputs['Alternative Dispute Resolution'] = inputs['Alternative Dispute Resolution'].map({'Yes': 1, 'No': 0})
            unmapped_adr = inputs['Alternative Dispute Resolution'].isnull().sum()
            if unmapped_adr > 0:
                mapping_errors['Alternative Dispute Resolution'] = unmapped_adr

        if 'Attorney/Representative' in inputs.columns:
            inputs['Attorney/Representative'] = inputs['Attorney/Representative'].map({'No': 0, 'Yes': 1})
            unmapped_attorney = inputs['Attorney/Representative'].isnull().sum()
            if unmapped_attorney > 0:
                mapping_errors['Attorney/Representative'] = unmapped_attorney

        if 'COVID-19 Indicator' in inputs.columns:
            inputs['COVID-19 Indicator'] = inputs['COVID-19 Indicator'].map({'No': 0, 'Yes': 1})
            unmapped_covid = inputs['COVID-19 Indicator'].isnull().sum()
            if unmapped_covid > 0:
                mapping_errors['COVID-19 Indicator'] = unmapped_covid
    except Exception as e:
        st.error(f"An error occurred during mapping: {e}")
        return

    # Check for mapping errors
    if mapping_errors:
        for col, count in mapping_errors.items():
            st.warning(f"{count} value(s) in the column '{col}' could not be mapped. Please check input data.")
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
