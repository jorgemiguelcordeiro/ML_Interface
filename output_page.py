
#Importing the libraries
import streamlit as st
import joblib
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
    # Define the zip file and expected CSV file
    zip_file = "train_data.zip"  # Ensure the zip file is in the same directory
    csv_file = "train_data.csv"

    # Extract the contents of the zip file
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            if csv_file in z.namelist():
                z.extractall()
                st.success(f"Successfully extracted '{csv_file}' from '{zip_file}'.")
            else:
                st.error(f"'{csv_file}' not found in '{zip_file}'.")
                return
    except FileNotFoundError:
        st.error(f"Zip file '{zip_file}' not found. Ensure it is in the correct directory.")
        return
    except zipfile.BadZipFile:
        st.error(f"'{zip_file}' is not a valid zip file.")
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
    except pd.errors.EmptyDataError:
        st.error(f"CSV file '{csv_file}' is empty.")
        return
    except pd.errors.ParserError:
        st.error(f"CSV file '{csv_file}' could not be parsed. Please ensure it is a valid CSV file.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        return

    # Set 'Claim Identifier' as the index for the dataset
    try:
        train_data.set_index('Claim Identifier', inplace=True)
        st.success("'Claim Identifier' set as the index.")
    except KeyError:
        st.error("'Claim Identifier' column not found in the dataset. Please verify the dataset structure.")
        return

    # Copy and process the dataset
    train_to_split = train_data.copy()

    # Handle columns of interest with all missing values
    columns_of_interest = train_to_split.columns[train_to_split.isnull().sum() == train_to_split.shape[0]]
    if columns_of_interest.empty:
        st.warning("No columns with all missing values found.")
    else:
        train_to_split = train_to_split.dropna(subset=columns_of_interest, how='all')
        st.info(f"Dropped rows where all columns in {columns_of_interest.tolist()} have NaN values.")

    # Drop unused columns
    try:
        train_to_split = train_to_split.drop(columns=['OIICS Nature of Injury Description'], errors='ignore')
        st.info("Dropped 'OIICS Nature of Injury Description' column.")
    except KeyError:
        st.warning("'OIICS Nature of Injury Description' column not found.")

    # Define features (X) by excluding target columns
    try:
        X = train_to_split.drop(columns=['Agreement Reached', 'WCB Decision', 'Claim Injury Type'], errors='ignore')
        st.success("Feature matrix (X) created by excluding target columns.")
        st.write("Feature Matrix:")
        st.dataframe(X.head())
    except KeyError as e:
        st.error(f"An error occurred while creating the feature matrix: {e}")
        return

    st.success("Data loading completed.")
    
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
     
    # Debug: Show raw input data
    st.write("**Raw Input Data Before Processing:**", inputs)


    #Preprocessing_pipeline

  

    def preprocessing_pipeline(train, val=None, test=None, outlier_treatment=True):
        debug_info = {}
        # Configurações gerais
        covid_start = pd.Timestamp('2020-03-01')
        covid_end = pd.Timestamp('2021-12-31')
    
        binary_columns = ['Attorney/Representative', 'COVID-19 Indicator']
        columns_to_replace = ['Age at Injury', 'Average Weekly Wage', 'Birth Year']
        description_columns = [
        ('Industry Code', 'Industry Code Description'),
        ('WCIO Cause of Injury Code', 'WCIO Cause of Injury Description'),
        ('WCIO Nature of Injury Code', 'WCIO Nature of Injury Description'),
        ('WCIO Part Of Body Code', 'WCIO Part Of Body Description')]
    
        variables_to_treat_outliers = ['Age at Injury']
        iqr_threshold = 1.5
    
        cols_to_impute = {
            'mode': [['Gender', 'Alternative Dispute Resolution', 'Industry Code Description', 'WCIO Nature of Injury Description', 
                         'WCIO Cause of Injury Description', 'WCIO Part Of Body Description']],
            'knn': ['Age at Injury']}
        columns_to_remove = [
            'First Hearing Date', 'C-3 Date', 'IME-4 Count', 'Average Weekly Wage', 'Birth Year', 
            'COVID-19 Indicator', 'Accident Date', 'Assembly Date', 'C-2 Date', 
            'Carrier Name', 'County of Injury', 'District Name', 'Zip Code', 'Industry Code',
            'WCIO Cause of Injury Code',  
            'WCIO Nature of Injury Code', 'WCIO Part Of Body Code', 'Industry Code Description', 
            'WCIO Cause of Injury Description', 'WCIO Nature of Injury Description', 
            'WCIO Part Of Body Description', 'Carrier Type']
    
        def debug_step(name, df):
            """Imprime informações úteis sobre o estado do dataframe."""
            #print(f"DEBUG [{name}] - Shape: {df.shape}, Nulls: {df.isnull().sum().sum()}")
            debug_info[name] = {"shape": df.shape, "null_count": df.isnull().sum().sum()}
        
        def convert_to_binary(df, columns):
            for col in columns:
                df[col] = (df[col] == 'Y').astype(int)
            debug_step("convert_to_binary", df)
            return df
    
        def convert_numeric_columns(df):
            numeric_cols = df.select_dtypes(['int64', 'float64']).columns
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            #debug_step("convert_numeric_columns", df)
            return df
    
        def convert_categorical_and_dates(df):
            for col in df.select_dtypes(['object', 'category']).columns:
                if 'Date' in col:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype('category')
            return df
            #debug_step("convert_categorical_and_dates", df)
    
        def add_covid_flag(df, covid_start, covid_end):
            df['COVID Period'] = ((df['Accident Date'] >= covid_start) & 
                                  (df['Accident Date'] <= covid_end)).astype(int)
            #debug_step("add_covid_flag", df)
            return df
    
        def winsorize_with_iqr(df, columns, iqr_threshold):
            bounds = {}
            for col in columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - iqr_threshold * iqr
                upper = q3 + iqr_threshold * iqr
                df[col] = df[col].clip(lower, upper)
                bounds[col] = (lower, upper)
                print(f"Bounds for {col}: Lower = {lower}, Upper = {upper}")
            #debug_step("winsorize_with_iqr", df)
            return df, bounds
            
        def process_missing_values(df, is_train=False, cols_to_impute=None, imputers=None, scalers=None):
          	"""
          	Processa os dados, realizando imputação no caso de treino e reutilizando objetos ajustados nos outros casos.
          	
          	Args:
          	    df (DataFrame): O dataframe para imputar dados.
          	    is_train (bool): Se estamos no conjunto de treino.
          	    cols_to_impute (dict): Estrutura com as colunas a imputar.
          	    imputers (dict): Dicionário para armazenar imputadores ajustados.
          	    scalers (dict): Dicionário para armazenar escaladores ajustados.
          	    
          	Returns:
          	    df (DataFrame): Dados processados.
          	    imputers (dict): Objetos ajustados para imputação.
          	    scalers (dict): Objetos ajustados para escalonamento.
          	"""
          	
          	# Se os dicionários não existirem, inicialize-os
          	if imputers is None:
          	    imputers = {'mode': {}, 'knn': {}}
          	if scalers is None:
          	    scalers = {}
          	
          	# Processamento para colunas 'mode' - imputando com 'most_frequent'
          	for col_group_list in cols_to_impute['mode']:
          	    for col_group in col_group_list:
          		if is_train:
          		    if col_group not in imputers['mode']:
          			imputers['mode'][col_group] = SimpleImputer(strategy='most_frequent')
          		    df[col_group] = imputers['mode'][col_group].fit_transform(df[[col_group]]).ravel()
          		else:
          		    if col_group in imputers['mode']:
          			df[col_group] = imputers['mode'][col_group].transform(df[[col_group]]).ravel()
              
          	# Processamento para colunas 'knn' com imputação usando KNN e escalonamento
          	for col in cols_to_impute['knn']:
          	    if is_train:
          		# Ajustar o imputador e escalador apenas no treinamento
          		scaler = StandardScaler()
          		knn_imputer = KNNImputer(n_neighbors=5)
              
          		# Aplicar escalonamento nos dados
          		df_scaled = scaler.fit_transform(df[[col]])
          		
          		# Aplicar imputação nos dados escalonados
          		df[col] = knn_imputer.fit_transform(df_scaled).ravel()
          		
          		# Salvar os objetos ajustados para uso nos outros conjuntos de dados
          		scalers[col] = scaler
          		imputers['knn'][col] = knn_imputer
          	    else:
          		# Usar apenas os objetos ajustados no treinamento para transformação
          		if col in scalers and col in imputers['knn']:
          		    df_scaled = scalers[col].transform(df[[col]])
          		    df[col] = imputers['knn'][col].transform(df_scaled).ravel()
          		    # Reverter o escalonamento após imputação com inverse_transform
          	    
          	for col in cols_to_impute['knn']:
          	    if col in scalers:
          		df[col] = scalers[col].inverse_transform(df[[col]]).ravel()
          	    
          	return df, imputers, scalers
          
          	

        
	    
        def process_gender_and_alternative_dispute(df):
            if 'Gender' in df.columns:
                df['Gender'] = np.where(df['Gender'].isin(['U', 'X']), np.nan, df['Gender'])
            if 'Alternative Dispute Resolution' in df.columns:
                df['Alternative Dispute Resolution'] = np.where(df['Alternative Dispute Resolution'] == 'U', 
                                                                np.nan, 
                                                                df['Alternative Dispute Resolution'])
            return df
            debug_step("process_gender_and_alternative_dispute", df)
    
        def categorize_delay_days(x):
            if pd.isna(x) or x < 0:
                return 'Invalid'
            elif x <= 30:
                return "Short delays (0-30 days)"
            elif x <= 180:
                return "Medium delays (31-180 days)"
            elif x <= 365:
                return "Long delays (181-365 days)"
            else:
                return "Very long delays (>365 days)"
            debug_step("categorize_delay_days", df)
    
        def categorize_missing_info(df):
            def categorize(x):
                if x == 0:
                    return "No missing information"
                elif x <= 2:
                    return "Low missing information"
                elif x <= 4:
                    return "Moderate missing information"
                elif x <= 8:
                    return "High missing information"
                else:
                    return "Very high missing information"
            
            df['missing_info_category'] = df.isna().sum(axis=1).apply(categorize)
            debug_step("categorize_missing_info", df)
            return df
    
        def categorize_wage(x):
            if pd.isna(x) or x <= 0:
                return 'Invalid'
            elif x <= 702:
                return 'Very Low Income'
            elif x <= 1100:
                return 'Low Income'
            elif x <= 1600:
                return 'Middle Income'
            elif x <= 3000:
                return 'Upper Middle Income (Q3 to Upper Fence)'
            else:
                return 'High Income (> Upper Fence)'
            debug_step("categorize_wage", df)
    
        def categorize_ime4_count(count):
            if pd.isna(count) or count < 0:
                return 'Invalid'
            elif count < 1:
                return "Low IME-4 Count"
            elif 1 <= count <= 2:
                return "Low IME-4 Count"
            elif 2 < count <= 4:
                return "Medium IME-4 Count"
            elif 4 < count <= 8.5:
                return "High IME-4 Count"
            else:
                return "Very High IME-4 Count"
            debug_step("categorize_ime4_count", df)
    
        def check_missing_dates(row):
            missing_columns = [
                col for col in ["Accident Date", "C-3 Date", "C-2 Date", "Assembly Date", "First Hearing Date"] 
                if pd.isna(row[col])]
            return ", ".join(missing_columns) if missing_columns else "OK"
            debug_step("check_missing_dates", df)
    
        def validate_dates(df):
            # Rule 1: Assembly Date before Accident Date
            df["Assembly Date before Accident Date"] = df.apply(lambda row: 
                1 if pd.notna(row["Accident Date"]) and pd.notna(row["Assembly Date"]) and row["Assembly Date"] < row["Accident Date"] 
                else 0, axis=1)
            
            # Rule 2: C-2 or C-3 Date before Accident Date
            df["C-2 or C-3 Date before Accident Date"] = df.apply(lambda row: 
                1 if pd.notna(row["Accident Date"]) and any([
                    pd.notna(row["C-2 Date"]) and row["C-2 Date"] < row["Accident Date"],
                    pd.notna(row["C-3 Date"]) and row["C-3 Date"] < row["Accident Date"]
                ]) 
                else 0, axis=1)
        
            # Rule 3: C-2 or C-3 Date after Assembly Date
            df["C-2 or C-3 Date after Assembly Date"] = df.apply(lambda row: 
                1 if pd.notna(row["Assembly Date"]) and any([
                    pd.notna(row["C-2 Date"]) and row["C-2 Date"] > row["Assembly Date"],
                    pd.notna(row["C-3 Date"]) and row["C-3 Date"] > row["Assembly Date"]
                ]) 
                else 0, axis=1)
            debug_step("validate_dates", df)
            return df
    
        # Funções de mapeamento para diferentes descrições
        def industry_map(description):
            for category, descriptions in industry_mapping.items():
                if description in descriptions:
                    return category
            return "Unmapped"
        
        def nature_of_injury_map(description):
            for category, descriptions in nature_of_injury_mapping.items():
                if description in descriptions:
                    return category
            return "Unmapped"
        
        def map_cause(description):
            for category, descriptions in WCIO_Cause_map.items():
                if description in descriptions:
                    return category
            return "Unmapped"
        
        def map_part_of_body(description):
            for category, descriptions in part_of_body_mapping.items():
                if description in descriptions:
                    return category
            return "Unmapped"
        
        def map_carrier_type(carrier):
            for category, carriers in carrier_type_mapping.items():
                if carrier in carriers:
                    return category
            return "Unmapped"
        
        def group_rare_categories(series, threshold=1000):
            counts = series.value_counts()
            rare_categories = counts[counts < threshold].index
            #print(f"Rare categories for {series.name}: {rare_categories.tolist()}")
            return series.replace(rare_categories, 'Other')
        
        def apply_mappings_and_grouping(train, val=None, test=None, mappings=None, thresholds=None):
            """
            Aplica mapeamentos e agrupamento de categorias raras nos conjuntos de dados.
            """
            # Aplicar mapeamentos nos dados de treinamento
            if mappings is None:
                mappings = {
                    'Industry Code Description': industry_map,
                    'WCIO Nature of Injury Description': nature_of_injury_map,
                    'WCIO Cause of Injury Description': map_cause,
                    'WCIO Part Of Body Description': map_part_of_body,
                    'Carrier Type': map_carrier_type
                }
        
                for column, map_func in mappings.items():
                    if column in train.columns:
                        train[f'Mapped {column}'] = train[column].apply(map_func)
                        #print(f"Mapped '{column}' created")
                    else:
                        print(f"Column '{column}' not found in train DataFrame")
        
            # Aplicar mapeamentos em validação e teste
            if val is not None:
                for column, map_func in mappings.items():
                    if column in val.columns:
                        val[f'Mapped {column}'] = val[column].apply(map_func)
                    else:
                        print(f"Column '{column}' not found in validation DataFrame")
        
            if test is not None:
                for column, map_func in mappings.items():
                    if column in test.columns:
                        test[f'Mapped {column}'] = test[column].apply(map_func)
                    else:
                        print(f"Column '{column}' not found in test DataFrame")
        
            # Retornar os datasets separadamente
            return train, val, test
    
        
        def process_dataset(df, is_train=False, mappings=None, bounds=None, covid_start=None, covid_end=None,
                            cols_to_impute=None, imputers=None, scalers=None):
            # Corrigir Tipos de Dados e Formatos
            df = convert_to_binary(df, binary_columns)
            df = convert_numeric_columns(df)
            df = convert_categorical_and_dates(df)
            df = process_gender_and_alternative_dispute(df)
            df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)
        
            # Engenharia de Recursos
            # Modificação: Definir 'Birth Year' < 1934 como 0
            df['Birth Year'] = df['Birth Year'].apply(lambda x: 0 if x < 1934 else x)
            df['Missing_Dates'] = df.apply(check_missing_dates, axis=1)
            df = validate_dates(df)
            df['delay_days_category'] = (df['Assembly Date'] - df['Accident Date']).dt.days.apply(categorize_delay_days)
            df = categorize_missing_info(df)
            df['Wage Category'] = df['Average Weekly Wage'].apply(categorize_wage)
            df['IME-4 Count Category'] = df['IME-4 Count'].apply(categorize_ime4_count)
            df = add_covid_flag(df, covid_start, covid_end)
            
            if is_train:
                # Processando no treinamento: ajustando imputadores e scalers
                df, imputers, scalers = process_missing_values(df, is_train=True, cols_to_impute=cols_to_impute, imputers=None, scalers=None)
                # Criando Mapeamentos
                mappings = {}
                for code_col, desc_col in description_columns:
                    mappings[desc_col] = df.groupby(desc_col)[code_col].apply(lambda x: x.value_counts().idxmax()).to_dict()
                    df[code_col] = df[desc_col].map(mappings[desc_col])
                    #print(f"Standardized {code_col} based on {desc_col}. Unique mappings created: {len(mappings[desc_col])}")
                
                if outlier_treatment:
                    df, bounds = winsorize_with_iqr(df, variables_to_treat_outliers, iqr_threshold)
                else:
                    bounds = {}  # Inicializar bounds como dicionário vazio
        
            else:
                # Processando dados fora do treinamento usando apenas imputadores e scalers ajustados
                if imputers is None or scalers is None:
                    raise ValueError("Imputers and scalers must be provided for validation and test datasets.")
                
                # Processando os dados apenas transformando-os
                df, _, _ = process_missing_values(df, is_train=False, cols_to_impute=cols_to_impute, imputers=imputers, scalers=scalers)
        
                # Aplicando os mapeamentos salvos durante o treinamento
                if mappings is None:
                    raise ValueError("Mappings must be provided for validation and test datasets.")
                
                for code_col, desc_col in description_columns:
                    df[code_col] = df[desc_col].map(mappings[desc_col])
                    #print(f"Applied training mappings to standardize {code_col}.")
    
                if outlier_treatment:
                    if bounds is None:
                        raise ValueError("Bounds must be provided for validation and test datasets.")
                    for col in variables_to_treat_outliers:
                        if col in bounds:
                            lower, upper = bounds[col]
                            df[col] = df[col].clip(lower, upper)
        
            return (df, imputers, scalers, mappings, bounds) if is_train else df
    
        # Process the training dataset
        train, imputers, scalers, mappings, bounds = process_dataset(
            train, is_train=True, covid_start=covid_start, covid_end=covid_end, cols_to_impute=cols_to_impute)
    
        # Process validation and test datasets
        if val is not None:
            val = process_dataset(val, is_train=False, mappings=mappings, bounds=bounds, covid_start=covid_start, 
                                  covid_end=covid_end, cols_to_impute=cols_to_impute, imputers=imputers, scalers=scalers)
    
        if test is not None:
            test = process_dataset(test, is_train=False, mappings=mappings, bounds=bounds, covid_start=covid_start, 
                                   covid_end=covid_end, cols_to_impute=cols_to_impute, imputers=imputers, scalers=scalers)
    
        train, val, test = apply_mappings_and_grouping(train, val=val, test=test)
    
        # Drop irrelevant columns
        train = train.drop(columns=columns_to_remove, errors='ignore')
        if val is not None:
            val = val.drop(columns=columns_to_remove, errors='ignore')
        if test is not None:
            test = test.drop(columns=columns_to_remove, errors='ignore')
            
        # Return the datasets
        if test is not None and val is not None:
            return train, val, test
        elif val is not None:
            return train, val
        elif test is not None:
            return train, test
        else:
            return train


    #applying the preprocessing pipeline function

    train_feature_selection_prep, user_unput = preprocessing_pipeline(train_to_split, inputs, test= None, outlier_treatment=True)
    # Pipeline 2: Grouping
    def map_region(row):
        region_group_mapping = {
            'I': 'Medical Group 1',
            'II': 'Medical Group 1',
            'III': 'Medical Group 1',
            'IV': 'Medical Group 2',
            'UK': 'UK'
        }
        return region_group_mapping.get(row, row)
    
    def group_c2_related(row):
        if "C-2" in row:
            return "C-2 Related"
        return row
    
    def group_delay_days(row):
        if row in ['Long delays (181-365 days)', 'Medium delays (31-180 days)']:
            return 'Grouped Delays (31-365 days)'
        return row
    
    def group_missing_info(row):
        if row in ['Low missing information', 'No missing information']:
            return 'Grouped Low/No Missing Info'
        elif row in ['High missing information', 'Moderate missing information']:
            return 'Grouped High/Moderate Missing Info'
        return row
    
    def group_wage_category(row):
        if row in ['High Income (> Upper Fence)', 'Middle Income', 'Upper Middle Income (Q3 to Upper Fence)']:
            return 'Grouped Middle to High Income'
        elif row in ['Low Income', 'Very Low Income']:
            return 'Grouped Low Income'
        return row
    
    def group_ime4_count(row):
        if row in ['High IME-4 Count', 'Very High IME-4 Count']:
            return 'Grouped High IME-4 Count'
        elif row in ['Low IME-4 Count', 'Medium IME-4 Count']:
            return 'Grouped Low/Medium IME-4 Count'
        return row
    
    def group_industry_code(row):
        if row in ['Financial & Business Activities', 'Public Service', 'Service-Providing Industries']:
            return 'Grouped Service-Providing Industries'
        elif row in ['Goods-Producing Industries', 'Trade, Transportation, and Utilities']:
            return 'Grouped Goods-Producing/Trade Industries'
        return row
    
    def group_injury_cause(row):
        if row in ['I. Burn or Scald – Heat or Cold', 'III. Cut, Puncture, Scrape Injured By', 'IX. Rubbed or Abraded By']:
            return 'Grouped Other Injuries'
        elif row in ['II. Caught In, Under or Between', 'VII. Striking Against or Stepping On', 'VIII.Struck or Injured By']:
            return 'Grouped Striking/Stepping Injuries'
        elif row in ['IV. Fall, Slip or Trip Injury', 'VI. Strain or Injury By']:
            return 'Grouped Fall/Strain Injuries'
        return row
    
    def group_part_body(row):
        if row in ['III. Upper Extremities', 'V. Lower Extremities']:
            return 'Grouped Extremities'
        elif row in ['IV. Trunk', 'OTHERS or Multiple Areas']:
            return 'Grouped Trunk/Other Areas'
        return row
    
    # Função principal para aplicar agrupamentos
    def apply_groupings(df):
        df['Alternative Dispute Resolution'] = df['Alternative Dispute Resolution'].map({'N': 0, 'Y': 1})
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
        df['Medical Fee Region'] = df['Medical Fee Region'].map(map_region)
        df["Missing_Dates"] = df["Missing_Dates"].apply(group_c2_related)
        df['delay_days_category'] = df['delay_days_category'].apply(group_delay_days)
        df['missing_info_category'] = df['missing_info_category'].apply(group_missing_info)
        df['Wage Category'] = df['Wage Category'].apply(group_wage_category)
        df['IME-4 Count Category'] = df['IME-4 Count Category'].apply(group_ime4_count)
        df['Mapped Industry Code Description'] = df['Mapped Industry Code Description'].apply(group_industry_code)
        df['Mapped WCIO Cause of Injury Description'] = df['Mapped WCIO Cause of Injury Description'].apply(group_injury_cause)
        df['Mapped WCIO Part Of Body Description'] = df['Mapped WCIO Part Of Body Description'].apply(group_part_body)
        
        return df
    train_feature_selection_group = apply_groupings(train_feature_selection_prep)

    columns_to_scale = ['Age at Injury', 'Number of Dependents']

    columns_already_encoded = [
        'Alternative Dispute Resolution',
        'Attorney/Representative',
        'Gender',
        'Assembly Date before Accident Date',
        'C-2 or C-3 Date before Accident Date',
        'C-2 or C-3 Date after Assembly Date',
        'COVID Period']
    
    ordinal_columns = ['delay_days_category', 'IME-4 Count Category', 'Wage Category', 'missing_info_category']
    
    one_hot_columns = [
        'Medical Fee Region', 'Missing_Dates', 'Mapped Industry Code Description', 'Mapped WCIO Nature of Injury Description',
        'Mapped WCIO Cause of Injury Description', 'Mapped WCIO Part Of Body Description','Mapped Carrier Type']
	

    def encoder_and_scaler(scaler, fit_data, df, ordinal_columns, one_hot_columns, numerical_columns):
        scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler()}
        scaler = scalers[scaler]
        
        # Encoders and scalers must be fitted on the train data
        ordinal_encoder = OrdinalEncoder()
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Use sparse_output=False
        
        # Fit encoders on the fit_data
        if ordinal_columns:
            ordinal_encoder.fit(fit_data[ordinal_columns])
        if one_hot_columns:
            one_hot_encoder.fit(fit_data[one_hot_columns])
        if numerical_columns:
            scaler.fit(fit_data[numerical_columns])
        
        # Apply transformations
        ordinal_encoded = ordinal_encoder.transform(df[ordinal_columns]) if ordinal_columns else None
        one_hot_encoded = one_hot_encoder.transform(df[one_hot_columns]) if one_hot_columns else None
        scaled_numerical = scaler.transform(df[numerical_columns]) if numerical_columns else None
        
        # Generate new column names
        one_hot_columns_names = (
            one_hot_encoder.get_feature_names_out(one_hot_columns) if one_hot_columns else []
        )
        
        # Combine transformed data into a DataFrame
        transformed_dfs = []
        if ordinal_encoded is not None:
            transformed_dfs.append(pd.DataFrame(ordinal_encoded, columns=ordinal_columns, index=df.index))
        if one_hot_encoded is not None:
            transformed_dfs.append(pd.DataFrame(one_hot_encoded, columns=one_hot_columns_names, index=df.index))
        if scaled_numerical is not None:
            transformed_dfs.append(pd.DataFrame(scaled_numerical, columns=numerical_columns, index=df.index))
        
        # Include any columns not transformed
        other_columns = df.drop(columns=ordinal_columns + one_hot_columns + numerical_columns, errors='ignore')
        if not other_columns.empty:
            transformed_dfs.append(other_columns)
        
        # Concatenate all parts into a single DataFrame
        transformed_df = pd.concat(transformed_dfs, axis=1)
        
        return transformed_df
    
        train_feature_selection = encoder_and_scaler(
            "standard", 
            train_feature_selection_group, 
            train_feature_selection_group, 
            ordinal_columns, 
            one_hot_columns, 
            columns_to_scale
        )
    
    # Fit and apply transformations within each fold
    X_train_fold, X_val_fold = preprocessing_pipeline(train_to_split, inputs, test=None, outlier_treatment=outlier_treatment)
    X_train_fold_to_scale = apply_groupings(X_train_fold)
    
    if scaler or ordinal_columns or one_hot_columns or numerical_columns:
        X_train_fold = encoder_and_scaler(
            scaler, 
            fit_data=X_train_fold_to_scale, 
            df=X_train_fold_to_scale,
            ordinal_columns=ordinal_columns,
            one_hot_columns=one_hot_columns,
            numerical_columns=numerical_columns
        )
    
    # Select features
    X_train_fold = X_train_fold[features]
    X_val_fold = X_val_fold[features]



  
		
    # Perform prediction
    try:
        prediction = model.predict(X_val_fold)[0]
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


