import streamlit as st
import datetime

# Existing dictionary mappings (provided by you)
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

# Additional dictionary-based mappings for fields that previously had hard-coded values
gender_mapping = {
    "Male": ["Male"],
    "Female": ["Female"]
}

adr_mapping = {
    "Yes": ["Yes"],
    "No": ["No"]
}

attorney_mapping = {
    "Yes": ["Yes"],
    "No": ["No"]
}

covid_mapping = {
    "Yes": ["Yes"],
    "No": ["No"]
}

# OIICS nature of injury description mapping (sample)
oiics_nature_of_injury_description_mapping = {
    "Fractures": ["Fractures"],
    "Sprains": ["Sprains"],
    "Cuts": ["Cuts"],
    "Burns": ["Burns"]
}

def get_allowed_values(mapping_dict):
    allowed_values = []
    for key, values in mapping_dict.items():
        cleaned_values = [v for v in values if v.strip() != '']
        allowed_values.extend(cleaned_values)
    # Remove duplicates and sort
    allowed_values = list(set(allowed_values))
    allowed_values.sort()
    return allowed_values

def input_page():
    # Sidebar branding/instructions
    st.sidebar.image("New York Workers' Compensation Board (WCB).png", use_column_width=True)  # Replace with your logo
    st.sidebar.write("**Instructions:**")
    st.sidebar.write("1. Fill out all the necessary claim-related information.")
    st.sidebar.write("2. Use the hints next to each field for clarification.")
    st.sidebar.write("3. Click **Submit** at the bottom when done.")
    st.sidebar.markdown("---")
    st.sidebar.write("For support, contact: support@company.com")

    # Page title and instructions at the top
    st.markdown(
        "<p style='color:red; font-weight:bold; font-size:16px;'>THIS FORM MAY ONLY BE SUBMITTED ELECTRONICALLY. DO NOT MAIL.</p>",
        unsafe_allow_html=True
    )
    st.title("Input Claim Data")
    st.write("Please provide the following information for the claim. Fields are organized for clarity. Hover over the info icons for additional details.")

    # Group 1: Dates & Basic Identifiers
    with st.expander("Basic Claim Details"):
        col1, col2 = st.columns(2)
        with col1:
            accident_date = st.date_input("Accident Date", value=datetime.date.today(), help="Date when the accident occurred.")
            assembly_date = st.date_input("Assembly Date", value=datetime.date.today(), help="Date the claim was first assembled by the board.")

        with col2:
            c2_date = st.date_input("C-2 Date", value=datetime.date.today(), help="Date the Employer's Report of Work-Related Injury/Illness (C-2) was received.")
            c3_date = st.date_input("C-3 Date", value=datetime.date.today(), help="Date Employee Claim Form (C-3) was received.")
            first_hearing_date = st.date_input("First Hearing Date", value=datetime.date.today(), help="Date of the first hearing on the claim.")

    current_year = datetime.date.today().year
    min_birth_year = current_year - 80

    # Group 2: Personal & Employment Details
    with st.expander("Personal & Employment Details"):
        col3, col4 = st.columns(2)
        with col3:
            age_at_injury = st.number_input("Age at Injury", min_value=10, max_value=80, value=30, help="Age of the injured worker at the time of injury.")
            birth_year = st.number_input("Birth Year", min_value=min_birth_year, max_value=current_year, value=min_birth_year, help="Year of birth of the injured worker.")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Gender of the injured worker.")
            average_weekly_wage = st.number_input("Average Weekly Wage", min_value=0.0, value=1000.0, help="Wage used to calculate benefits.")

        with col4:
            attorney_representative = st.selectbox("Attorney/Representative", ["Yes", "No"], help="Whether the claim has an attorney or representative.")
            alternative_dispute_resolution = st.selectbox("Alternative Dispute Resolution", ["Yes", "No"], help="Whether external adjudication processes are used.")

    # Group 3: Location & Industry Details
    with st.expander("Location & Industry"):
        col5, col6 = st.columns(2)
        with col5:
            counties = ["Albany", "Bronx", "Brooklyn", "Queens", "Richmond", "Manhattan", "Suffolk", "Nassau", "Westchester"]
            county_of_injury = st.selectbox("County of Injury", counties, help="New York county where the injury occurred.")
            district_name = st.selectbox("District Name", ["Albany", "Binghamton", "Brooklyn", "Buffalo", "Long Island", "Manhattan", "Queens", "Rochester", "Syracuse"], help="WCB district office overseeing the claim.")
            zip_code = st.text_input("Zip Code", placeholder="e.g., 10001", help="ZIP code of the injured worker’s home address.")

        with col6:
            medical_fee_regions = ["Upstate", "Downstate", "NYC"]
            medical_fee_region = st.selectbox("Medical Fee Region", medical_fee_regions, help="Region where the injured worker receives medical service.")
            industry_code = st.text_input("Industry Code", placeholder="e.g., 561320", help="NAICS code for the employer's industry.")
            industry_code_description = st.text_input("Industry Code Description", placeholder="e.g., Construction", help="Industry description of the employer.")

    # Group 4: Injury Specifics
    with st.expander("Injury Specifics"):
        col7, col8 = st.columns(2)
        with col7:
            covid_19_indicator = st.selectbox("COVID-19 Indicator", ["Yes", "No"], help="Indicate if the claim may be associated with COVID-19.")
            ime4_count = st.number_input("IME-4 Count", min_value=0, value=0, help="Number of IME-4 forms received (Independent Examiner's Report).")
            wcio_nature_of_injury_description = st.text_input("WCIO Nature of Injury Description", placeholder="e.g., Fracture", help="Description of the nature of injury.")

        with col8:
            wcio_cause_of_injury_code = st.text_input("WCIO Cause of Injury Code", placeholder="e.g., 03", help="WCIO code representing the cause of injury.")
            wcio_cause_of_injury_description = st.text_input("WCIO Cause of Injury Description", placeholder="e.g., Fall", help="Description of the cause of injury.")
            wcio_part_of_body_code = st.text_input("WCIO Part Of Body Code", placeholder="e.g., 15", help="WCIO code for affected part of body.")
            wcio_part_of_body_description = st.text_input("WCIO Part Of Body Description", placeholder="e.g., Head", help="Description of the affected body part.")

    # Group 5: Carrier & Claim Details
    with st.expander("Carrier & Claim"):
        carrier_name = st.text_input("Carrier Name", placeholder="e.g., ABC Insurance", help="Primary insurance provider responsible for coverage.")
        carrier_type = st.selectbox("Carrier Type", ["Insurance Company", "Self-Insured Employer", "Group Self-Insurer", "State Insurance Fund"], help="Type of insurance provider.")

    # Group 6: Additional Details
    with st.expander("Additional Claim Details"):
        agreement_reached = st.selectbox("Agreement Reached", ["Yes", "No"], help="Indicates if an agreement was reached without WCB involvement.")
        wcb_decision = st.selectbox("WCB Decision", ["Accident", "Occupational Disease", "Pending"], help="Decision category relative to the claim by WCB.")
        number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0, help="Number of dependents claimed in the case.")

    # Collect all inputs into a dictionary
    inputs = {
        'Accident Date': accident_date,
        'Age at Injury': age_at_injury,
        'Alternative Dispute Resolution': alternative_dispute_resolution,
        'Assembly Date': assembly_date,
        'Attorney/Representative': attorney_representative,
        'Average Weekly Wage': average_weekly_wage,
        'Birth Year': birth_year,
        'C-2 Date': c2_date,
        'C-3 Date': c3_date,
        'Carrier Name': carrier_name,
        'Carrier Type': carrier_type,
        'County of Injury': county_of_injury,
        'COVID-19 Indicator': covid_19_indicator,
        'District Name': district_name,
        'First Hearing Date': first_hearing_date,
        'Gender': gender,
        'IME-4 Count': ime4_count,
        'Industry Code': industry_code,
        'Industry Code Description': industry_code_description,
        'Medical Fee Region': medical_fee_region,
        'WCIO Cause of Injury Code': wcio_cause_of_injury_code,
        'WCIO Cause of Injury Description': wcio_cause_of_injury_description,
        'WCIO Nature of Injury Description': wcio_nature_of_injury_description,
        'WCIO Part Of Body Code': wcio_part_of_body_code,
        'WCIO Part Of Body Description': wcio_part_of_body_description,
        'Zip Code': zip_code,
        'Agreement Reached': agreement_reached,
        'WCB Decision': wcb_decision,
        'Number of Dependents': number_of_dependents,
    }

    st.markdown("**Please review all the information carefully before submitting.**")
    if st.button("Submit"):
        st.session_state.inputs = inputs
        st.session_state.page = "output"
        st.success("Your data has been submitted! You will be redirected to the output page.")


