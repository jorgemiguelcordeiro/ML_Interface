import logging
logging.basicConfig(level=logging.DEBUG)

import joblib
import zipfile
import io
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
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
import time
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              BaggingClassifier, VotingClassifier, GradientBoostingClassifier, 
                              StackingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import streamlit as st


#this codes are from NAICS - https://www.bls.gov/iag/tgs/iag_index_naics.htm
industry_mapping = {
    'Industry_high_0': ['MANAGEMENT OF COMPANIES AND ENTERPRISES'],
    'Industry_high_1': [
        'HEALTH CARE AND SOCIAL ASSISTANCE', 'MANUFACTURING', 'FINANCE AND INSURANCE',
        'REAL ESTATE AND RENTAL AND LEASING', 'OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)',
        'RETAIL TRADE', 'ACCOMMODATION AND FOOD SERVICES', 'MINING',
        'PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES', 'ARTS, ENTERTAINMENT, AND RECREATION'],
    'Industry_high_2': ['EDUCATIONAL SERVICES'],
    'Industry_high_3_mid_5': [
        'CONSTRUCTION', 'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT',
        'TRANSPORTATION AND WAREHOUSING', 'WHOLESALE TRADE'],
    'Industry_high_4': ['UTILITIES', 'PUBLIC ADMINISTRATION'],
    'Industry_high_5': ['INFORMATION'],
    'Industry_high_3_2': ['AGRICULTURE, FORESTRY, FISHING AND HUNTING']
}

nature_of_injury_mapping = {
    'Nature of Injury Cluster 10': ['BYSSINOSIS', 'BLACK LUNG', 'VDT - RELATED DISEASES'],
    'Nature of Injury Cluster 4': ['ENUCLEATION', 'HERNIA', 'MENTAL DISORDER'],
    'Nature of Injury Cluster 17': ['DISLOCATION'],
    'Nature of Injury Cluster 15': ['FRACTURE', 'SEVERANCE', 'CRUSHING', 'CARPAL TUNNEL SYNDROME'],
    'Nature of Injury Cluster 1': ['AMPUTATION'],
    'Nature of Injury Cluster 9': ['RUPTURE'],
    'Nature of Injury Cluster 6': ['SILICOSIS'],
    'Nature of Injury Cluster 16': ['MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL', 'MENTAL STRESS',
                                'ALL OTHER SPECIFIC INJURIES, NOC', 'INFLAMMATION', 'ALL OTHER CUMULATIVE INJURY, NOC', 'CONTUSION'],
    'Nature of Injury Cluster 12': ['SPRAIN OR TEAR', 'CONCUSSION', 'MULTIPLE PHYSICAL INJURIES ONLY', 'STRAIN OR TEAR', 'DUST DISEASE, NOC'],
    'Nature of Injury Cluster 0': ['ASBESTOSIS'],
    'Nature of Injury Cluster 2': ['COVID-19', 'VISION LOSS', 'FREEZING', 'BURN', 'AIDS', 'ELECTRIC SHOCK', 'INFECTION', 'LACERATION',
                               'POISONING - CHEMICAL, (OTHER THAN METALS)', 'RESPIRATORY DISORDERS', 'FOREIGN BODY', 'HEAT PROSTRATION',
                               'POISONING - GENERAL (NOT OD OR CUMULATIVE'],
    'Nature of Injury Cluster 14': ['ASPHYXIATION'],
    'Nature of Injury Cluster 8': ['VASCULAR'],
    'Nature of Injury Cluster 5': ['ANGINA PECTORIS', 'POISONING - METAL', 'MYOCARDIAL INFARCTION',
                               'ADVERSE REACTION TO A VACCINATION OR INOCULATION', 'DERMATITIS', 'NO PHYSICAL INJURY',
                               'CONTAGIOUS DISEASE', 'SYNCOPE', 'PUNCTURE', 'RADIATION'],
    'Nature of Injury Cluster 3': ['CANCER'],
    'Nature of Injury Cluster 13': ['ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC'],
    'Nature of Injury Cluster 7': ['LOSS OF HEARING', 'HEARING LOSS OR IMPAIRMENT'],
    'Nature of Injury Cluster 11': ['HEPATITIS C']
}

#https://www.guarantysupport.com/wp-content/uploads/2024/02/WCIO-Legacy.pdf & https://www.mwcia.org/Media/Default/PDF/NewsFeed/Circulars/21-1787.pdf
WCIO_Cause_map = {
    'Cause of Injury Cluster 11': ['CRASH OF RAIL VEHICLE'],
    'Cause of Injury Cluster 15': ['FROM LADDER OR SCAFFOLDING', 'MOTOR VEHICLE', 'VEHICLE UPSET',
                               'COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE', 'MOTOR VEHICLE, NOC'],
    'Cause of Injury Cluster 10': ['GUNSHOT'],
    'Cause of Injury Cluster 0': ['COLLAPSING MATERIALS (SLIDES OF EARTH)', 'MOVING PARTS OF MACHINE',
                              'JUMPING OR LEAPING', 'MACHINE OR MACHINERY', 'SLIP, OR TRIP, DID NOT FALL',
                              'STRAIN OR INJURY BY, NOC', 'MOVING PART OF MACHINE', 'WIELDING OR THROWING'],
    'Cause of Injury Cluster 14': ['INTO OPENINGS', 'LIFTING', 'ON STAIRS', 'CRASH OF AIRPLANE', 'HOLDING OR CARRYING',
                               'TWISTING', 'FALL, SLIP OR TRIP, NOC', 'FROM LIQUID OR GREASE SPILLS', 'ON SAME LEVEL',
                               'COLLISION WITH A FIXED OBJECT', 'ON ICE OR SNOW'],
    'Cause of Injury Cluster 4': ['FROM DIFFERENT LEVEL (ELEVATION)', 'REACHING', 'USING TOOL OR MACHINERY'],
    'Cause of Injury Cluster 8': ['EXPLOSION OR FLARE BACK', 'CRASH OF WATER VEHICLE', 'FIRE OR FLAME', 
                              'POWERED HAND TOOL, APPLIANCE', 'SANDING, SCRAPING, CLEANING OPERATION',
                              'FALLING OR FLYING OBJECT', 'CAUGHT IN, UNDER OR BETWEEN, NOC', 'HAND TOOL OR MACHINE IN USE',
                              'OBJECT HANDLED', 'STRIKING AGAINST OR STEPPING ON, NOC', 'FELLOW WORKER, PATIENT OR OTHER PERSON',
                              'OBJECT HANDLED BY OTHERS', 'ELECTRICAL CURRENT', 'STRUCK OR INJURED, NOC', 'STATIONARY OBJECT',
                              'STEAM OR HOT FLUIDS', 'WELDING OPERATION', 'OBJECT BEING LIFTED OR HANDLED'],
    'Cause of Injury Cluster 12': ['PUSHING OR PULLING', 'PERSON IN ACT OF A CRIME'],
    'Cause of Injury Cluster 6': ['REPETITIVE MOTION', 'CUMULATIVE, NOC'],
    'Cause of Injury Cluster 13': ['PANDEMIC', 'OTHER - MISCELLANEOUS, NOC', 'OTHER THAN PHYSICAL CAUSE OF INJURY',
                               'ABSORPTION, INGESTION OR INHALATION, NOC'],
    'Cause of Injury Cluster 3': ['BROKEN GLASS', 'HOT OBJECTS OR SUBSTANCES', 'TEMPERATURE EXTREMES', 'STEPPING ON SHARP OBJECT',
                              'CHEMICALS', 'COLD OBJECTS OR SUBSTANCES', 'RUBBED OR ABRADED, NOC', 'CONTACT WITH, NOC',
                              'ANIMAL OR INSECT', 'CUT, PUNCTURE, SCRAPE, NOC', 'HAND TOOL, UTENSIL; NOT POWERED',
                              'RADIATION', 'FOREIGN MATTER (BODY) IN EYE(S)', 'MOLD'],
    'Cause of Injury Cluster 5': ['TERRORISM'],
    'Cause of Injury Cluster 9': ['CONTINUAL NOISE'],
    'Cause of Injury Cluster 7': ['DUST, GASES, FUMES OR VAPORS'],
    'Cause of Injury Cluster 2': ['ABNORMAL AIR PRESSURE'],
    'Cause of Injury Cluster 1': ['NATURAL DISASTERS']
}

part_of_body_mapping = {
    'Part Of Body Cluster 3': ['WHOLE BODY', 'HEART'],
    'Part Of Body Cluster 10': ['INTERNAL ORGANS'],
    'Part Of Body Cluster 9': ['LUNGS', 'BODY SYSTEMS AND MULTIPLE BODY SYSTEMS', 'NO PHYSICAL INJURY'],
    'Part Of Body Cluster 6': ['BRAIN', 'SPINAL CORD'],
    'Part Of Body Cluster 7': ['INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED', 'WRIST (S) & HAND(S)'],
    'Part Of Body Cluster 4': ['CHEST', 'SKULL', 'BUTTOCKS', 'SOFT TISSUE', 'MULTIPLE HEAD INJURY',
                           'ABDOMEN INCLUDING GROIN', 'LOWER LEG', 'LOWER ARM', 'HAND', 'FACIAL BONES',
                           'FOOT', 'GREAT TOE', 'MULTIPLE LOWER EXTREMITIES', 'NOSE', 'SACRUM AND COCCYX', 'TOES', 'UPPER LEG'],
    'Part Of Body Cluster 2': ['MULTIPLE BODY PARTS (INCLUDING BODY', 'MULTIPLE NECK INJURY', 'LOWER BACK AREA',
                           'LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA', 'UPPER BACK AREA', 'DISC', 'MULTIPLE TRUNK',
                           'PELVIS', 'VERTEBRAE'],
    'Part Of Body Cluster 1': ['MOUTH', 'EYE(S)', 'FINGER(S)', 'LARYNX', 'TEETH', 'THUMB'],
    'Part Of Body Cluster 12': ['MULTIPLE', 'KNEE', 'ELBOW', 'ANKLE', 'SHOULDER(S)', 'MULTIPLE UPPER EXTREMITIES', 'WRIST'],
    'Part Of Body Cluster 0': ['UPPER ARM', 'HIP'],
    'Part Of Body Cluster 5': ['ARTIFICIAL APPLIANCE'],
    'Part Of Body Cluster 8': ['EAR(S)'],
    'Part Of Body Cluster 11': ['TRACHEA']
}

# Dictionary to map 'Carrier Type' 
carrier_type_mapping = {
    "Private Carriers": ["1A. PRIVATE", "4A. SELF PRIVATE"],
    "Public Carriers": ["3A. SELF PUBLIC", "2A. SIF"],
    "SF_0_3": ["5C. SPECIAL FUND - POI CARRIER WCB MENANDS"],
    "SF_0_2": ["5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)"],
    "SF_2_1": ["5D. SPECIAL FUND - UNKNOWN"],
    "Unknown": ["UNKNOWN"]
}

columns_to_scale = ['Age at Injury']

columns_already_encoded = [
    'Alternative Dispute Resolution',
    'Attorney/Representative',
    'Gender',
    'Assembly Date before Accident Date',
    'Assembly Date or C-2 or C-3 Date before Accident Date',
    'COVID Period'
]

ordinal_columns = ['delay_days_category', 'IME-4 Count Category', 'Wage Category', 'missing_info_category','Number of Dependents']

one_hot_columns = [
    'Medical Fee Region', 'Missing_Dates', 'Mapped Industry Code Description', 'Mapped WCIO Nature of Injury Description',
    'Mapped WCIO Cause of Injury Description', 'Mapped WCIO Part Of Body Description','Mapped Carrier Type'
]


def preprocessing_pipeline(train, val=None, test=None, outlier_treatment=True):
    """Pipeline that handles data cleaning & feature engineering for train, val, test data."""
    debug_info = {}
    # Geral configurations
    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2021-12-31')

    binary_columns = ['Attorney/Representative', 'COVID-19 Indicator']
    columns_to_replace = ['Age at Injury', 'Average Weekly Wage', 'Birth Year']
    description_columns = [
        ('Industry Code', 'Industry Code Description'),
        ('WCIO Cause of Injury Code', 'WCIO Cause of Injury Description'),
        ('WCIO Nature of Injury Code', 'WCIO Nature of Injury Description'),
        ('WCIO Part Of Body Code', 'WCIO Part Of Body Description')
    ]

    variables_to_treat_outliers = ['Age at Injury']
    iqr_threshold = 1.5

    cols_to_impute = {
        'mode': [['Gender', 'Alternative Dispute Resolution', 'Industry Code Description', 'WCIO Nature of Injury Description', 
                     'WCIO Cause of Injury Description', 'WCIO Part Of Body Description']],
        'knn': ['Age at Injury']
    }
    columns_to_remove = [
        'First Hearing Date', 'C-3 Date', 'IME-4 Count', 'Average Weekly Wage', 'Birth Year', 
        'COVID-19 Indicator', 'Accident Date', 'Assembly Date', 'C-2 Date', 
        'Carrier Name', 'County of Injury', 'District Name', 'Zip Code', 'Industry Code',
        'WCIO Cause of Injury Code',  
        'WCIO Nature of Injury Code', 'WCIO Part Of Body Code', 'Industry Code Description', 
        'WCIO Cause of Injury Description', 'WCIO Nature of Injury Description', 
        'WCIO Part Of Body Description', 'Carrier Type'
    ]

    def debug_step(name, df):
        """Captures shape and null count for debug."""
        debug_info[name] = {"shape": df.shape, "null_count": df.isnull().sum().sum()}

    def convert_to_binary(df, columns):
        for col in columns:
            df[col] = (df[col] == 'Y').astype(int)
        debug_step("convert_to_binary", df)
        return df

    def convert_numeric_columns(df):
        numeric_cols = df.select_dtypes(['int64', 'float64']).columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        return df

    def convert_categorical_and_dates(df):
        for col in df.select_dtypes(['object', 'category']).columns:
            if 'Date' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = df[col].astype('category')
        return df

    def add_covid_flag(df, covid_start, covid_end):
        df['COVID Period'] = ((df['Accident Date'] >= covid_start) & 
                              (df['Accident Date'] <= covid_end)).astype(int)
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
        return df, bounds
        
    def process_missing_values(df, is_train=False, cols_to_impute=None, imputers=None, scalers=None):
        if imputers is None:
            imputers = {'mode': {}, 'knn': {}}
        if scalers is None:
            scalers = {}
        
        # imputing 'mode' with 'most_frequent'
        for col_group_list in cols_to_impute['mode']:
            for col_group in col_group_list:
                if is_train:
                    if col_group not in imputers['mode']:
                        imputers['mode'][col_group] = SimpleImputer(strategy='most_frequent')
                    df[col_group] = imputers['mode'][col_group].fit_transform(df[[col_group]]).ravel()
                else:
                    if col_group in imputers['mode']:
                        df[col_group] = imputers['mode'][col_group].transform(df[[col_group]]).ravel()
    
        # KNN 
        for col in cols_to_impute['knn']:
            if is_train:
                # train data only
                scaler = StandardScaler()
                knn_imputer = KNNImputer(n_neighbors=5)
                df_scaled = scaler.fit_transform(df[[col]])
                df[col] = knn_imputer.fit_transform(df_scaled).ravel()
                scalers[col] = scaler
                imputers['knn'][col] = knn_imputer
            else:
                # based on the train imputers and scalers
                if col in scalers and col in imputers['knn']:
                    df_scaled = scalers[col].transform(df[[col]])
                    df[col] = imputers['knn'][col].transform(df_scaled).ravel()
            
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

    def check_missing_dates(row):
        missing_columns = [
            col for col in ["Accident Date", "C-3 Date", "C-2 Date", "Assembly Date", "First Hearing Date"] 
            if pd.isna(row[col])]
        return ", ".join(missing_columns) if missing_columns else "OK"

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
        return series.replace(rare_categories, 'Other')
    
    def apply_mappings_and_grouping(train, val=None, test=None, mappings=None, thresholds=None):
        """Aplica mapeamentos e agrupamento de categorias raras nos conjuntos de dados."""
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
                else:
                    print(f"Column '{column}' not found in train DataFrame")
    
        if val is not None:
            for column, map_func in mappings.items():
                if column in val.columns:
                    val[f'Mapped {column}'] = val[column].apply(map_func)
    
        if test is not None:
            for column, map_func in mappings.items():
                if column in test.columns:
                    test[f'Mapped {column}'] = test[column].apply(map_func)
    
        return train, val, test

    def process_dataset(df, is_train=False, mappings=None, bounds=None, covid_start=None, covid_end=None,
                        cols_to_impute=None, imputers=None, scalers=None):
        df = convert_to_binary(df, binary_columns)
        df = convert_numeric_columns(df)
        df = convert_categorical_and_dates(df)
        df = process_gender_and_alternative_dispute(df)
        df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)
    
        # Feature Engineering
        df['Birth Year'] = df['Birth Year'].apply(lambda x: 0 if x < 1934 else x)
        df['Missing_Dates'] = df.apply(check_missing_dates, axis=1)
        df = validate_dates(df)
        df['delay_days_category'] = (df['Assembly Date'] - df['Accident Date']).dt.days.apply(categorize_delay_days)
        df = categorize_missing_info(df)
        df['Wage Category'] = df['Average Weekly Wage'].apply(categorize_wage)
        df['IME-4 Count Category'] = df['IME-4 Count'].apply(categorize_ime4_count)
        df = add_covid_flag(df, covid_start, covid_end)
        
        if is_train:
            df, imputers, scalers = process_missing_values(df, is_train=True, cols_to_impute=cols_to_impute, imputers=None, scalers=None)
            mappings = {}
            for code_col, desc_col in description_columns:
                mappings[desc_col] = df.groupby(desc_col)[code_col].apply(lambda x: x.value_counts().idxmax()).to_dict()
                df[code_col] = df[desc_col].map(mappings[desc_col])
            if outlier_treatment:
                df, bounds = winsorize_with_iqr(df, variables_to_treat_outliers, iqr_threshold)
            else:
                bounds = {}
    
        else:
            if imputers is None or scalers is None:
                raise ValueError("Imputers and scalers must be provided for validation and test datasets.")
            df, _, _ = process_missing_values(df, is_train=False, cols_to_impute=cols_to_impute, imputers=imputers, scalers=scalers)
            if mappings is None:
                raise ValueError("Mappings must be provided for validation and test datasets.")
            for code_col, desc_col in description_columns:
                df[code_col] = df[desc_col].map(mappings[desc_col])
            if outlier_treatment:
                if bounds is None:
                    raise ValueError("Bounds must be provided for validation and test datasets.")
                for col in variables_to_treat_outliers:
                    if col in bounds:
                        lower, upper = bounds[col]
                        df[col] = df[col].clip(lower, upper)
    
        return (df, imputers, scalers, mappings, bounds) if is_train else df

    train, imputers, scalers, mappings, bounds = process_dataset(
        train, is_train=True, covid_start=covid_start, covid_end=covid_end, cols_to_impute=cols_to_impute
    )

    if val is not None:
        val = process_dataset(val, is_train=False, mappings=mappings, bounds=bounds, covid_start=covid_start, 
                              covid_end=covid_end, cols_to_impute=cols_to_impute, imputers=imputers, scalers=scalers)

    if test is not None:
        test = process_dataset(test, is_train=False, mappings=mappings, bounds=bounds, covid_start=covid_start, 
                               covid_end=covid_end, cols_to_impute=cols_to_impute, imputers=imputers, scalers=scalers)

    train, val, test = apply_mappings_and_grouping(train, val=val, test=test)
    train = train.drop(columns=columns_to_remove, errors='ignore')
    if val is not None:
        val = val.drop(columns=columns_to_remove, errors='ignore')
    if test is not None:
        test = test.drop(columns=columns_to_remove, errors='ignore')
        
    if test is not None and val is not None:
        return train, val, test
    elif val is not None:
        return train, val
    elif test is not None:
        return train, test
    else:
        return train


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
    elif row in ['Accident Date', 'Accident Date, C-3 Date']:
        return 'Accident Date, C-3 Date'
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
    if row in ['Industry_high_1', 'Industry_high_2']:
        return 'Industry_high_1_2'
    elif row in ['Industry_high_4', 'Industry_high_5']:
        return 'Industry_high_4_5'
    elif row in ['Industry_high_3_2', 'Industry_high_3_mid_5']:
        return 'Industry_high_3_2'
    return row

def group_injury_cause(row):
    if row in ['Cause of Injury Cluster 0', 'Cause of Injury Cluster 14']:
        return 'Cause of Injury Cluster 0_14'
    elif row in ['Cause of Injury Cluster 12', 'Cause of Injury Cluster 6']:
        return 'Cause of Injury Cluster 12_6'
    elif row in ['Cause of Injury Cluster 7', 'Cause of Injury Cluster 8']:
        return 'Cause of Injury Cluster 7_8'
    return row

def group_part_body(row):
    if row in ['Part Of Body Cluster 0', 'Part Of Body Cluster 12']:
        return 'Part Of Body Cluster 0_12'
    return row

def apply_groupings(df):
    df['Alternative Dispute Resolution'] = df['Alternative Dispute Resolution'].replace({'N': 0, 'Y': 1}).astype('int32')
    df['Gender'] = df['Gender'].replace({'F': 0, 'M': 1}).astype('int32')
    df['Medical Fee Region'] = df['Medical Fee Region'].map(map_region)
    df["Missing_Dates"] = df["Missing_Dates"].apply(group_c2_related)
    df['delay_days_category'] = df['delay_days_category'].apply(group_delay_days)
    df['missing_info_category'] = df['missing_info_category'].apply(group_missing_info)
    df['Wage Category'] = df['Wage Category'].apply(group_wage_category)
    df['IME-4 Count Category'] = df['IME-4 Count Category'].apply(group_ime4_count)
    df['Mapped Industry Code Description'] = df['Mapped Industry Code Description'].apply(group_industry_code)
    df['Mapped WCIO Cause of Injury Description'] = df['Mapped WCIO Cause of Injury Description'].apply(group_injury_cause)
    df['Mapped WCIO Part Of Body Description'] = df['Mapped WCIO Part Of Body Description'].apply(group_part_body)
    df['Assembly Date or C-2 or C-3 Date before Accident Date'] = (
        df['Assembly Date before Accident Date'] | df['C-2 or C-3 Date before Accident Date']
    )
    df = df.drop(columns=['Assembly Date before Accident Date', 'C-2 or C-3 Date before Accident Date'])
    return df

def encoder_and_scaler(scaler, fit_data, df, ordinal_columns, one_hot_columns, numerical_columns):
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler()}
    scaler = scalers[scaler]
    
    ordinal_encoder = OrdinalEncoder()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    if ordinal_columns:
        ordinal_encoder.fit(fit_data[ordinal_columns])
    if one_hot_columns:
        one_hot_encoder.fit(fit_data[one_hot_columns])
    if numerical_columns:
        scaler.fit(fit_data[numerical_columns])
    
    ordinal_encoded = ordinal_encoder.transform(df[ordinal_columns]) if ordinal_columns else None
    one_hot_encoded = one_hot_encoder.transform(df[one_hot_columns]) if one_hot_columns else None
    scaled_numerical = scaler.transform(df[numerical_columns]) if numerical_columns else None
    
    one_hot_columns_names = (
        one_hot_encoder.get_feature_names_out(one_hot_columns) if one_hot_columns else []
    )
    
    transformed_dfs = []
    if ordinal_encoded is not None:
        transformed_dfs.append(pd.DataFrame(ordinal_encoded, columns=ordinal_columns, index=df.index))
    if one_hot_encoded is not None:
        transformed_dfs.append(pd.DataFrame(one_hot_encoded, columns=one_hot_columns_names, index=df.index))
    if scaled_numerical is not None:
        transformed_dfs.append(pd.DataFrame(scaled_numerical, columns=numerical_columns, index=df.index))
    
    other_columns = df.drop(columns=ordinal_columns + one_hot_columns + numerical_columns, errors='ignore')
    if not other_columns.empty:
        transformed_dfs.append(other_columns)
    
    transformed_df = pd.concat(transformed_dfs, axis=1)
    return transformed_df

def reduce_memory_usage(df, continuous_columns):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()
        
        if col_type != object and col_type.name != 'category':
            if col in continuous_columns:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            else:
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# (placeholder) This is an example feature list
# Remove "Number of Dependents" from the features list
features = ['delay_days_category', 'IME-4 Count Category', 'Wage Category', 'missing_info_category', 'Medical Fee Region_Medical Group 1', 'Medical Fee Region_UK', 'Missing_Dates_Accident Date, C-3 Date', 'Missing_Dates_Accident Date, C-3 Date, First Hearing Date', 'Missing_Dates_Accident Date, First Hearing Date', 'Missing_Dates_C-2 Related', 'Missing_Dates_C-3 Date', 'Missing_Dates_C-3 Date, First Hearing Date', 'Missing_Dates_First Hearing Date', 'Missing_Dates_OK', 'Mapped Industry Code Description_Industry_high_0', 'Mapped Industry Code Description_Industry_high_1_2', 'Mapped Industry Code Description_Industry_high_3_2', 'Mapped Industry Code Description_Industry_high_4_5', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 0', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 1', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 10', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 11', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 12', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 13', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 14', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 15', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 16', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 17', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 2', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 3', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 4', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 5', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 6', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 8', 'Mapped WCIO Nature of Injury Description_Nature of Injury Cluster 9', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 0_14', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 1', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 10', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 11', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 12_6', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 13', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 15', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 2', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 3', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 4', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 5', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 7_8', 'Mapped WCIO Cause of Injury Description_Cause of Injury Cluster 9', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 0_12', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 1', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 10', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 11', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 2', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 3', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 4', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 5', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 6', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 7', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 8', 'Mapped WCIO Part Of Body Description_Part Of Body Cluster 9', 'Mapped Carrier Type_Public Carriers', 'Mapped Carrier Type_SF_0_2', 'Mapped Carrier Type_SF_0_3', 'Mapped Carrier Type_SF_2_1', 'Mapped Carrier Type_Unknown', 'Age at Injury', 'Alternative Dispute Resolution', 'Attorney/Representative', 'Gender', 'C-2 or C-3 Date after Assembly Date', 'COVID Period', 'Assembly Date or C-2 or C-3 Date before Accident Date']

# Here we assume you have a local CSV with your training data
with zipfile.ZipFile("train_data.zip", 'r') as z:
    with z.open("train_data.csv") as f:
        train_data = pd.read_csv(f)  
train_data.set_index('Claim Identifier', inplace=True)
columns_of_interest = train_data.columns[train_data.isnull().sum() == 19445]
train_to_split = train_data.dropna(subset=columns_of_interest, how='all')
train_to_split = train_to_split.drop(columns = 'OIICS Nature of Injury Description', errors='ignore')
target_features = ["Agreement Reached", "WCB Decision", "Claim Injury Type"]
X = train_to_split.drop(columns=target_features)
y = train_to_split[['Claim Injury Type']]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.7, stratify=y, shuffle=True, random_state=42
)

# Load the trained model
model = joblib.load('optimized_logistic_2.joblib')  # Replace with the path to your trained model

# Columns that we want to treat as ordinal, one-hot, numerical
ordinal_columns = ['delay_days_category', 'IME-4 Count Category', 'Wage Category', 'missing_info_category']
one_hot_columns = [
    'Medical Fee Region', 'Missing_Dates',
    'Mapped Industry Code Description', 
    'Mapped WCIO Nature of Injury Description',
    'Mapped WCIO Cause of Injury Description', 
    'Mapped WCIO Part Of Body Description',
    'Mapped Carrier Type'
]
numerical_columns = columns_to_scale


def preprocess_input(train, interface_data, features,
                     ordinal_columns, one_hot_columns,
                     numerical_columns, outlier_treatment=True):
    """
    Preprocess user input to match the training data format.
    """
    train, interface_data = preprocessing_pipeline(
        train, test=interface_data, val=None, outlier_treatment=outlier_treatment
    )
    train_to_scale = apply_groupings(train)
    interface_data_to_scale = apply_groupings(interface_data)

    def encode_and_scale(data_to_fit, data, method="standard"):
        return encoder_and_scaler(
            method,
            fit_data=data_to_fit,
            df=data,
            ordinal_columns=ordinal_columns,
            one_hot_columns=one_hot_columns,
            numerical_columns=numerical_columns
        )

    # Fit transforms on train, apply to train
    train = encode_and_scale(train_to_scale, train_to_scale)
    # Apply transforms based on train to interface data
    interface_data = encode_and_scale(train_to_scale, interface_data_to_scale)

    train = reduce_memory_usage(train, numerical_columns)
    interface_data = reduce_memory_usage(interface_data, numerical_columns)

    # Select only the features needed for modeling
    train = train[features]
    interface_data = interface_data[features]
    return interface_data

def extract_column_info(train_data, excluded_columns):
    """
    Extracts unique values and formats for each column in the dataset.
    """
    column_info = {}
    for col in train_data.columns:
        if col not in excluded_columns:
            if train_data[col].dtype == 'object':
                column_info[col] = train_data[col].dropna().unique().tolist()
            elif np.issubdtype(train_data[col].dtype, np.number):
                column_info[col] = (train_data[col].min(), train_data[col].max())
            elif np.issubdtype(train_data[col].dtype, np.datetime64):
                column_info[col] = "Date format (YYYY-MM-DD)"
    return column_info


def input_page():
    """
    Streamlit page that collects input from the user, preprocesses it, and predicts the claim outcome.
    """
    st.sidebar.title("Instructions")
    st.sidebar.write(
        """Please provide details about the claim in the fields below. 
        Make sure each field is filled in according to the best available data. 
        You can leave a field blank if you don’t have any information for it."""
    )

    # Optionally, replace 'ny_logo.png' with your actual local file or an online URL:
    st.sidebar.image('New York Workers Compensation Board (WCB).png')

    # --- Main page content ---
    st.title("New York Workers’ Compensation Claim Prediction")
    # We build a dictionary of possible input options from the training data (for demonstration).
    # In practice, adapt to your use case: 
    # - Some columns might not make sense for user input, or
    # - You might have domain constraints you can enforce with Streamlit's UI widgets.
    column_info = extract_column_info(train_data, target_features)

    st.write("## Enter Claim Details")
    input_data = {}

    # For each column in column_info, create a suitable input in Streamlit
    for col, info in column_info.items():
        # We only want columns that appear in X_train (not target columns)
        if col not in X_train.columns:
            continue

        # Decide what kind of input widget to render, based on data type
        if isinstance(info, list):
            # Categorical: pick from a set of values
            input_data[col] = st.selectbox(f"{col}", [""] + info)
        elif isinstance(info, tuple):
            # Numerical data: get the min/max to help set defaults
            min_val, max_val = info
            # Convert to float if needed, or int
            # For demonstration, we'll assume numeric = float
            input_data[col] = st.number_input(
                f"{col}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(min_val) if min_val is not None else 0.0,
                step=1.0
            )
        elif info == "Date format (YYYY-MM-DD)":
            # Date input
            input_data[col] = st.date_input(f"{col}")
        else:
            # Fallback to text input
            input_data[col] = st.text_input(f"{col}")

    if st.button("Predict Outcome"):
        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        try:
            preprocessed_data = preprocess_input(
                train=X_train,
                interface_data=input_df,
                features=features,
                ordinal_columns=ordinal_columns,
                one_hot_columns=one_hot_columns,
                numerical_columns=numerical_columns,
                outlier_treatment=True
            )
            
            # Make prediction
            prediction = model.predict(preprocessed_data)
            st.success(f"The predicted outcome for the claim is: {prediction[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


# Streamlit apps typically execute top-to-bottom; 
# you can just call `input_page()` at the bottom of the script:
if __name__ == "__main__":
    input_page()


