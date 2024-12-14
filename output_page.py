
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



def output_page():
  
    st.title("Prediction Result")

    # === SECTION 1: Data loading ==
  
    # Extract the contents of the zip file
    zip_file = "train_data.zip"  # Ensure the zip file is in the same directory
    csv_file = "train_data.csv"
    try:
        z = zipfile.ZipFile(zip_file)
        z.extractall()
        del z
        st.success(f"Successfully extracted '{zip_file}'.")
    except FileNotFoundError:
        st.error(f"Zip file '{zip_file}' not found. Ensure it is in the correct directory.")
        return
    except Exception as e:
        st.error(f"An error occurred while extracting '{zip_file}': {e}")
        return

    # Load the dataset
    try:
        train_data = pd.read_csv(csv_file)
        st.success("Dataset loaded successfully.")
        st.write(f"Number of rows: {train_data.shape[0]}, Number of columns: {train_data.shape[1]}")
        st.write("Sample Data:")
        st.dataframe(train_data.head())  # Use Streamlit's dataframe widget for a cleaner display
    except FileNotFoundError:
        st.error(f"CSV file '{csv_file}' not found after extraction. Please verify the zip contents.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        return
      
    # Set Claim Identifier as the index for both datasets
    train_data.set_index('Claim Identifier', inplace=True)
    train_to_split = train_data.copy()

    columns_of_interest = train_to_split.columns[train_to_split.isnull().sum() == 19445]
    # Drop rows where all columns in columns_of_interest have NaN values
    train_to_split = train_to_split.dropna(subset=columns_of_interest, how='all')
    train_to_split = train_to_split.drop(columns = 'OIICS Nature of Injury Description')

    X = train_to_split.drop(columns= ['Agreement Reached','WCB Decision', 'Claim Injury Type')


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

    # Define the required columns based on your updated dataset
    required_columns = [
        'Gender', 'Alternative Dispute Resolution', 
        'Attorney/Representative', 'COVID-19 Indicator', 
        'Industry Code Description', 'WCIO Nature of Injury Description',
        'WCIO Cause of Injury Description', 'WCIO Part Of Body Description', 'Carrier Name'
    ]

    # Check for missing columns
    missing_required_cols = [col for col in required_columns if col not in inputs.columns]
    if missing_required_cols:
        for col in missing_required_cols:
            st.warning(f"'{col}' column is missing from the inputs. Please ensure it is captured in the input page.")

    # Drop unused columns (example)
    unused_columns = ['OIICS Nature of Injury Description']  # adjust or remove if needed
    inputs = inputs.drop(columns=unused_columns, errors='ignore')

    # Map categorical values (Gender, etc.)
    gender_map = {'Female': 0, 'Male': 1}
    if 'Gender' in inputs.columns:
        inputs['Gender'] = inputs['Gender'].map(gender_map)

    covid_map = {'No': 0, 'Yes': 1}
    if 'COVID-19 Indicator' in inputs.columns:
        inputs['COVID-19 Indicator'] = inputs['COVID-19 Indicator'].map(covid_map)

    adr_map = {'Yes': 1, 'No': 0}
    if 'Alternative Dispute Resolution' in inputs.columns:
        inputs['Alternative Dispute Resolution'] = inputs['Alternative Dispute Resolution'].map(adr_map)

    attorney_map = {'No': 0, 'Yes': 1}
    if 'Attorney/Representative' in inputs.columns:
        inputs['Attorney/Representative'] = inputs['Attorney/Representative'].map(attorney_map)

    # Apply dictionary-based category mappings
    if 'Industry Code Description' in inputs.columns:
        inputs['Industry Code Description'] = inputs['Industry Code Description'].apply(
            lambda x: map_value_to_category(x, industry_code_description_mapping)
        )

    if 'WCIO Nature of Injury Description' in inputs.columns:
        inputs['WCIO Nature of Injury Description'] = inputs['WCIO Nature of Injury Description'].apply(
            lambda x: map_value_to_category(x, wcio_nature_of_injury_description_mapping)
        )

    if 'WCIO Cause of Injury Description' in inputs.columns:
        inputs['WCIO Cause of Injury Description'] = inputs['WCIO Cause of Injury Description'].apply(
            lambda x: map_value_to_category(x, wcio_cause_of_injury_description_mapping)
        )

    if 'WCIO Part Of Body Description' in inputs.columns:
        inputs['WCIO Part Of Body Description'] = inputs['WCIO Part Of Body Description'].apply(
            lambda x: map_value_to_category(x, wcio_part_of_body_description_mapping)
        )

    if 'Carrier Name' in inputs.columns:
        inputs['Carrier Name'] = inputs['Carrier Name'].apply(
            lambda x: map_value_to_category(x, carrier_type_mapping)
        )

    st.write("**Processed Input Data After Mapping:**", inputs)

    # Check column alignment with the model
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


