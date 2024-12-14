
#Importing the libraries
import zipfile, io
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Define your mapping dictionaries outside the function
industry_code_description_mapping = {
    "Service-Providing Industries": [
        'ARTS, ENTERTAINMENT, AND RECREATION', 'ACCOMMODATION AND FOOD SERVICES', 'INFORMATION', 
        'OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)'
    ],
    "Trade, Transportation, and Utilities": [
        'WHOLESALE TRADE', 'RETAIL TRADE', 'TRANSPORTATION AND WAREHOUSING', 'UTILITIES'
    ],
    "Public Service": [
        'PUBLIC ADMINISTRATION', 'HEALTH CARE AND SOCIAL ASSISTANCE', 'EDUCATIONAL SERVICES'
    ],
    "Financial & Business Activities": [
        'FINANCE AND INSURANCE', 'REAL ESTATE AND RENTAL AND LEASING', 'MANAGEMENT OF COMPANIES AND ENTERPRISES',
        'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT', 'PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES'
    ],
    "Goods-Producing Industries": [
        'AGRICULTURE, FORESTRY, FISHING AND HUNTING','MINING','CONSTRUCTION','MANUFACTURING','','','',''
    ]
}

wcio_nature_of_injury_description_mapping = {
    "I. Specific Injury": [
        'NO PHYSICAL INJURY', 'AMPUTATION', 'ANGINA PECTORIS','BURN','CONCUSSION','CONTUSION',
        'CRUSHING','DISLOCATION','ELECTRIC SHOCK','ENUCLEATION','FOREIGN BODY','FRACTURE',
        'FREEZING','HEARING LOSS OR IMPAIRMENT','HEAT PROSTRATION','HERNIA','INFECTION',
        'INFLAMMATION','ADVERSE REACTION TO A VACCINATION OR INOCULATION','LACERATION',
        'MYOCARDIAL INFARCTION','POISONING - GENERAL (NOT OD OR CUMULATIVE','PUNCTURE','RUPTURE',
        'SEVERANCE','SPRAIN OR TEAR','STRAIN OR TEAR','SYNCOPE','ASPHYXIATION','VASCULAR',
        'VISION LOSS','ALL OTHER SPECIFIC INJURIES, NOC'
    ],
    "II. Occupational Disease or Cumulative Injury": [
        'DUST DISEASE, NOC', 'ASBESTOSIS','BLACK LUNG','BYSSINOSIS',
        'SILICOSIS','RESPIRATORY DISORDERS','POISONING - CHEMICAL, (OTHER THAN METALS)',
        'POISONING - METAL','DERMATITIS','MENTAL DISORDER','RADIATION',
        'ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC','LOSS OF HEARING','CONTAGIOUS DISEASE',
        'CANCER','AIDS','VDT - RELATED DISEASES','MENTAL STRESS','CARPAL TUNNEL SYNDROME',
        'HEPATITIS C','ALL OTHER CUMULATIVE INJURY, NOC','COVID-19'
    ],
    "III. Multiple Injuries": [
        'MULTIPLE PHYSICAL INJURIES ONLY', 'MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL'
    ]
}

wcio_cause_of_injury_description_mapping = {
    "I. Burn or Scald – Heat or Cold": [
        'CHEMICALS','HOT OBJECTS OR SUBSTANCES','TEMPERATURE EXTREMES','FIRE OR FLAME',
        'STEAM OR HOT FLUIDS','DUST, GASES, FUMES OR VAPORS','WELDING OPERATION',
        'RADIATION','CONTACT WITH, NOC','COLD OBJECTS OR SUBSTANCES','ABNORMAL AIR PRESSURE',
        'ELECTRICAL CURRENT'
    ],
    "II. Caught In, Under or Between": [
        'MACHINE OR MACHINERY','OBJECT HANDLED','CAUGHT IN, UNDER OR BETWEEN, NOC',
        'COLLAPSING MATERIALS (SLIDES OF EARTH)'
    ],
    "III. Cut, Puncture, Scrape Injured By": [
        'BROKEN GLASS','HAND TOOL, UTENSIL; NOT POWERED','OBJECT BEING LIFTED OR HANDLED',
        'POWERED HAND TOOL, APPLIANCE','CUT, PUNCTURE, SCRAPE, NOC'
    ],
    "IV. Fall, Slip or Trip Injury": [
        'FROM DIFFERENT LEVEL (ELEVATION)','FROM LADDER OR SCAFFOLDING',
        'FROM LIQUID OR GREASE SPILLS','INTO OPENINGS','ON SAME LEVEL','SLIP, OR TRIP, DID NOT FALL',
        'FALL, SLIP OR TRIP, NOC','ON ICE OR SNOW','ON STAIRS'
    ],
    "V. Motor Vehicle": [
        'CRASH OF WATER VEHICLE','CRASH OF RAIL VEHICLE','COLLISION WITH A FIXED OBJECT',
        'COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE','CRASH OF AIRPLANE','VEHICLE UPSET','MOTOR VEHICLE, NOC'
    ],
    "VI. Strain or Injury By": [
        'CONTINUAL NOISE','TWISTING','JUMPING OR LEAPING','HOLDING OR CARRYING','LIFTING',
        'PUSHING OR PULLING','REACHING','USING TOOL OR MACHINERY','STRAIN OR INJURY BY, NOC',
        'WIELDING OR THROWING','REPETITIVE MOTION'
    ],
    "VII. Striking Against or Stepping On": [
        'MOVING PART OF MACHINE','SANDING, SCRAPING, CLEANING OPERATION',
        'STATIONARY OBJECT','STEPPING ON SHARP OBJECT','STRIKING AGAINST OR STEPPING ON, NOC'
    ],
    "VIII.Struck or Injured By": [
        'FELLOW WORKER, PATIENT OR OTHER PERSON','FALLING OR FLYING OBJECT','HAND TOOL OR MACHINE IN USE',
        'MOTOR VEHICLE','MOVING PARTS OF MACHINE','OBJECT HANDLED BY OTHERS','STRUCK OR INJURED, NOC',
        'ANIMAL OR INSECT','EXPLOSION OR FLARE BACK'
    ],
    "IX. Rubbed or Abraded By": [
        'RUBBED OR ABRADED, NOC'
    ],
    "X. Miscellaneous Causes": [
        'ABSORPTION, INGESTION OR INHALATION, NOC','FOREIGN MATTER (BODY) IN EYE(S)','NATURAL DISASTERS',
        'PERSON IN ACT OF A CRIME','OTHER THAN PHYSICAL CAUSE OF INJURY','MOLD','TERRORISM',
        'CUMULATIVE, NOC','OTHER - MISCELLANEOUS, NOC','GUNSHOT','PANDEMIC'
    ]
}

wcio_part_of_body_description_mapping = {
    "OTHERS or Multiple Areas": [
        'ARTIFICIAL APPLIANCE', 'BODY SYSTEMS AND MULTIPLE BODY SYSTEMS',
        'INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED','MULTIPLE','MULTIPLE BODY PARTS (INCLUDING BODY',
        'NO PHYSICAL INJURY','WHOLE BODY', 'SOFT TISSUE'
    ],
    "I. Head": [
        'MULTIPLE HEAD INJURY', 'SKULL','BRAIN','EAR(S)','EYE(S)','NOSE','TEETH','MOUTH','FACIAL BONES'
    ],
    "II. Neck": [
        'MULTIPLE NECK INJURY','VERTEBRAE','DISC','SPINAL CORD','LARYNX','TRACHEA'
    ],
    "III. Upper Extremities": [
        'MULTIPLE UPPER EXTREMITIES','UPPER ARM','ELBOW','LOWER ARM','WRIST','WRIST (S) & HAND(S)', 'SHOULDER(S)',
        'HAND', 'THUMB', 'FINGER(S)'
    ],
    "IV. Trunk": [
        'MULTIPLE TRUNK','UPPER BACK AREA','LOWER BACK AREA','CHEST','SACRUM AND COCCYX','PELVIS','INTERNAL ORGANS',
        'HEART','LUNGS','ABDOMEN INCLUDING GROIN','BUTTOCKS','LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA',
        'LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA NOC TRUNK)'
    ],
    "V. Lower Extremities": [
        'MULTIPLE LOWER EXTREMITIES','HIP','UPPER LEG','KNEE','LOWER LEG','ANKLE','FOOT','TOES','GREAT TOE'
    ]
} 

carrier_type_mapping = {
    "Private Carriers": [
        "1A. PRIVATE", "4A. SELF PRIVATE"
    ],
    "Public Carriers": [
        "3A. SELF PUBLIC", "2A. SIF"
    ],
    "Special Funds": [
        "5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)", "5C. SPECIAL FUND - POI CARRIER WCB MENANDS",
        "5D. SPECIAL FUND - UNKNOWN"
    ],
    "Unknown": [
        "UNKNOWN"
    ]
}


def map_value_to_category(value, mapping_dict):
    if pd.isnull(value):
        return None
    # Convert both to uppercase for case-insensitive match
    val_upper = str(value).strip().upper()
    for key, values in mapping_dict.items():
        # Remove empty strings from values if any
        cleaned_values = [v.upper() for v in values if v.strip() != '']
        if val_upper in cleaned_values:
            return key
    return None

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



def output_page():
  
    st.title("Prediction Result")

    # === SECTION 1: Data loading ==
  
    
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


