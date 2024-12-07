
pip install streamlit
pip install pandas
pip install shap
pip install joblib
pip install dice-ml


import streamlit as st
import pandas as pd
import pickle
import os
import shap
import joblib
import dice_ml
from dice_ml import Dice
import streamlit.components.v1 as components

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def provide_explanation(user_reason):
    # Load model and preprocessor again (or could be passed as parameters if pre-loaded)
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')

    # Retrieve input data and prediction from session state
    input_data = st.session_state['input_data']
    prediction = st.session_state['prediction']

    input_df = pd.DataFrame([input_data])

    # Preprocess input data
    processed_input = preprocessor.transform(input_df)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_input)

    # Display SHAP force plot
    st.write("### Here's why the model made this prediction:")
    shap.initjs()
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], processed_input))

    # Generate counterfactual explanations using Dice
    st.write("### To get your expected outcome, consider the following changes:")

    # Load training data and define features
    # Adjust 'training_data.csv', 'feature1', 'feature2', and 'target' as per your data
    training_data = pd.read_csv('training_data.csv')
    continuous_features = ['feature1', 'feature2']  # Adjust as needed
    target = 'target'  # Adjust as needed

    dice_data = dice_ml.Data(
        dataframe=training_data,
        continuous_features=continuous_features,
        outcome_name=target
    )
    dice_model = dice_ml.Model(model=model, backend='sklearn')

    dice_exp = Dice(dice_data, dice_model, method='random')

    e1 = dice_exp.generate_counterfactuals(input_df, total_CFs=1, desired_class="opposite")
    st.write(e1.cf_examples_list[0].final_cfs_df)

    # Save the feedback (optional)
    feedback_data = {
        'input_data': input_data,
        'prediction': prediction,
        'expected_outcome': user_reason
    }
    with open('user_feedback.csv', 'a') as f:
        f.write(f"{feedback_data}\n")
    st.success("Thank you for your feedback!")

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

    # Store input_data and prediction in session_state for further use
    st.session_state['input_data'] = st.session_state.inputs
    st.session_state['prediction'] = prediction

    # Ask if the result was expected
    st.write("**Is this the result you were expecting?**")
    feedback = st.radio("", ('Yes', 'No'))

    if feedback == 'No':
        reason = st.text_area("Please explain why you expected a different result:")
        if st.button("Submit Feedback"):
            provide_explanation(reason)

    # Save option
    if st.button("Save Result"):
        result_df = pd.DataFrame([{'Prediction': prediction, **st.session_state.inputs}])
        result_df.to_csv('prediction_result.csv', index=False)
        st.success("Result saved as 'prediction_result.csv'")

    # Return to Welcome Page
    if st.button("Return to Welcome Page"):
        st.session_state.page = 'welcome'


