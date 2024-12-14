
#Importing the libraries
import zipfile, io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC  
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB


import warnings
from imblearn.over_sampling import SMOTE

# === SECTION 1: Data loading ===

z = zipfile.ZipFile("train_data.zip") #we loaded the project data folder zip in the same
#environment and use this code to extract the components
z.extractall()
del z
# Load datasets
train_data = pd.read_csv("train_data.csv")
print('Train df shape:', train_data.shape)

def output_page():
    st.title("Prediction Result")

    # Load the model from a .joblib file
    model_path = 'logistic_model.joblib'
    try:
        model = joblib.load(model_path)  # Use joblib.load instead of pickle.load
        st.write("Model loaded successfully.")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'logistic_model.joblib' is in the correct directory.")
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
    
    st.write("**Raw Input Data Before Processing:**", inputs)

    # Example of required columns
    required_columns = [
        'gender', 'alternative_dispute_resolution', 
        'attorney_representative', 'covid_19_indicator', 
        'industry_code_description', 'wcio_nature_of_injury_description',
        'wcio_cause_of_injury_description', 'wcio_part_of_body_description', 'carrier_name'
    ]
    
    missing_required_cols = [col for col in required_columns if col not in inputs.columns]

    if missing_required_cols:
        for col in missing_required_cols:
            st.warning(f"'{col}' column is missing from the inputs. Please ensure it is captured in the input page.")

    # Drop unused columns (example)
    unused_columns = ['OIICS Nature of Injury Description']  # adjust or remove if needed
    inputs = inputs.drop(columns=unused_columns, errors='ignore')

    # Map categorical values (gender, etc.)
    gender_map = {'Female': 0, 'Male': 1}
    if 'gender' in inputs.columns:
        inputs['gender'] = inputs['gender'].map(gender_map)

    covid_map = {'No': 0, 'Yes': 1}
    if 'covid_19_indicator' in inputs.columns:
        inputs['covid_19_indicator'] = inputs['covid_19_indicator'].map(covid_map)

    adr_map = {'Yes': 1, 'No': 0}
    if 'alternative_dispute_resolution' in inputs.columns:
        inputs['alternative_dispute_resolution'] = inputs['alternative_dispute_resolution'].map(adr_map)

    attorney_map = {'No': 0, 'Yes': 1}
    if 'attorney_representative' in inputs.columns:
        inputs['attorney_representative'] = inputs['attorney_representative'].map(attorney_map)

    # Apply dictionary-based category mappings
    if 'industry_code_description' in inputs.columns:
        inputs['industry_code_description'] = inputs['industry_code_description'].apply(
            lambda x: map_value_to_category(x, industry_code_description_mapping)
        )

    if 'wcio_nature_of_injury_description' in inputs.columns:
        inputs['wcio_nature_of_injury_description'] = inputs['wcio_nature_of_injury_description'].apply(
            lambda x: map_value_to_category(x, wcio_nature_of_injury_description_mapping)
        )

    if 'wcio_cause_of_injury_description' in inputs.columns:
        inputs['wcio_cause_of_injury_description'] = inputs['wcio_cause_of_injury_description'].apply(
            lambda x: map_value_to_category(x, wcio_cause_of_injury_description_mapping)
        )

    if 'wcio_part_of_body_description' in inputs.columns:
        inputs['wcio_part_of_body_description'] = inputs['wcio_part_of_body_description'].apply(
            lambda x: map_value_to_category(x, wcio_part_of_body_description_mapping)
        )

    if 'carrier_name' in inputs.columns:
        inputs['carrier_name'] = inputs['carrier_name'].apply(
            lambda x: map_value_to_category(x, carrier_type_mapping)
        )

    #st.write("**Processed Input Data Before Categorization:**", inputs)

    # Apply wage and IME-4 count categories if columns exist
    if 'average_weekly_wage' in inputs.columns:
        inputs['average_weekly_wage'] = inputs['average_weekly_wage'].apply(categorize_wage)
    else:
        st.warning("'average_weekly_wage' column not found. Unable to categorize wage.")

    if 'ime4_count' in inputs.columns:
        inputs['ime4_count'] = inputs['ime4_count'].apply(categorize_ime4_count)
    else:
        st.warning("'ime4_count' column not found. Unable to categorize IME-4 count.")

    st.write("**Processed Input Data After Categorization:**", inputs)




    
    # Check column alignment
    expected_columns = model.feature_names_in_  # or load from a saved list if needed
    current_columns = inputs.columns

    missing_columns = [col for col in expected_columns if col not in current_columns]
    extra_columns = [col for col in current_columns if col not in expected_columns]

    if missing_columns:
        st.write("The following columns were expected by the model but are missing in the input data:")
        for col in missing_columns:
            st.write(f"  - {col}")
        st.write("These missing columns indicate that you haven't reproduced the training preprocessing steps.")

    if extra_columns:
        st.write("The following columns are present in the input but were not seen during training:")
        for col in extra_columns:
            st.write(f"  - {col}")
        st.write("These extra columns suggest that you're providing features not seen by the model during training.")

    # If everything looks correct, proceed to prediction
    prediction = model.predict(inputs)[0]

    # Perform prediction
    try:
        prediction = model.predict(inputs)[0]
        st.subheader(f"The predicted Claim Injury Type is: **{prediction}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

    # Save Results
    if st.button("Save Result"):
        result_df = pd.DataFrame([{'Prediction': prediction, **st.session_state.inputs}])
        result_df.to_csv('prediction_result.csv', index=False)
        st.success("Result saved as 'prediction_result.csv'")

    # Navigation
    if st.button("Return to Welcome Page"):
        st.session_state.page = 'welcome'



