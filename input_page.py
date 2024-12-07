import streamlit as st
import datetime

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
    st.title("Input Claim Data")
    st.write("Please provide the following information for the claim. Fields are organized for clarity. Hover over the info icons for additional details.")

    # Group 1: Dates & Basic Identifiers
    with st.expander("Basic Claim Details"):
        # Use columns to arrange fields side-by-side
        col1, col2 = st.columns(2)

        with col1:
            accident_date = st.date_input("Accident Date", value=datetime.date.today(), help="Date when the accident occurred.")
            injury_date = st.date_input("Injury Date", value=datetime.date.today(), help="Date of the reported injury. Often same as Accident Date.")
            assembly_date = st.date_input("Assembly Date", value=datetime.date.today(), help="Date the claim was first assembled by the board.")
            claim_identifier = st.text_input("Claim Identifier", placeholder="e.g., CLM-12345", help="Unique ID assigned by WCB.")

        with col2:
            c2_date = st.date_input("C-2 Date", value=datetime.date.today(), help="Date the Employer's Report of Work-Related Injury/Illness (C-2) was received.")
            c3_date = st.date_input("C-3 Date", value=datetime.date.today(), help="Date Employee Claim Form (C-3) was received.")
            first_hearing_date = st.date_input("First Hearing Date", value=datetime.date.today(), help="Date of the first hearing on the claim.")

    # Group 2: Personal & Employment Details
    with st.expander("Personal & Employment Details"):
        col3, col4 = st.columns(2)

        with col3:
            age_at_injury = st.number_input("Age at Injury", min_value=0, max_value=100, value=30, help="Age of the injured worker at the time of injury.")
            birth_year = st.number_input("Birth Year", min_value=1900, max_value=datetime.date.today().year, value=1990, help="Year of birth of the injured worker.")
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], help="Gender of the injured worker.")
            average_weekly_wage = st.number_input("Average Weekly Wage ($)", min_value=0.0, value=1000.0, help="Wage used to calculate benefits.")

        with col4:
            attorney_representative = st.selectbox("Attorney/Representative", ["Yes", "No"], help="Whether the claim has an attorney or representative.")
            alternative_dispute_resolution = st.selectbox("Alternative Dispute Resolution", ["Yes", "No"], help="Whether external adjudication processes are used.")
            agreement_reached = st.selectbox("Agreement Reached", ["Yes", "No"], help="Indicates if an agreement was reached without WCB involvement.")
            wcb_decision = st.selectbox("WCB Decision", ["Accident", "Occupational Disease", "Pending"], help="Decision category relative to the claim by WCB.")

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
            industry_code_descriptions = ["Agriculture", "Construction", "Manufacturing", "Retail Trade", "Healthcare"]
            industry_code_description = st.selectbox("Industry Code Description", industry_code_descriptions, help="2-digit NAICS description of the employer's industry.")

    # Group 4: Injury Specifics
    with st.expander("Injury Specifics"):
        col7, col8 = st.columns(2)

        with col7:
            covid_19_indicator = st.selectbox("COVID-19 Indicator", ["Yes", "No"], help="Indicate if the claim may be associated with COVID-19.")
            ime4_count = st.number_input("IME-4 Count", min_value=0, value=0, help="Number of IME-4 forms received (Independent Examiner's Report).")
            oiics_nature_descriptions = ["Fractures", "Sprains", "Cuts", "Burns"]
            oiics_nature_of_injury_description = st.selectbox("OIICS Nature of Injury Description", oiics_nature_descriptions, help="OIICS code describing the nature of injury.")

        with col8:
            wcio_cause_of_injury_code = st.text_input("WCIO Cause of Injury Code", placeholder="e.g., 03", help="WCIO code representing the cause of injury.")
            wcio_cause_descriptions = ["Fall", "Struck by object", "Motor vehicle accident"]
            wcio_cause_of_injury_description = st.selectbox("WCIO Cause of Injury Description", wcio_cause_descriptions, help="Description of the cause of injury.")
            wcio_nature_of_injury_code = st.text_input("WCIO Nature of Injury Code", placeholder="e.g., 30", help="WCIO code representing the nature of injury.")
            wcio_nature_descriptions = ["Fracture", "Laceration", "Burn"]
            wcio_nature_of_injury_description = st.selectbox("WCIO Nature of Injury Description", wcio_nature_descriptions, help="Description of the nature of injury.")
            wcio_part_of_body_code = st.text_input("WCIO Part Of Body Code", placeholder="e.g., 15", help="WCIO code for affected part of body.")
            wcio_part_of_body_descriptions = ["Head", "Back", "Arm", "Leg"]
            wcio_part_of_body_description = st.selectbox("WCIO Part Of Body Description", wcio_part_of_body_descriptions, help="Description of the affected body part.")

    # Group 5: Carrier & Claim Details
    with st.expander("Carrier & Claim"):
        carrier_name = st.text_input("Carrier Name", placeholder="e.g., ABC Insurance", help="Primary insurance provider responsible for coverage.")
        carrier_type = st.selectbox("Carrier Type", ["Insurance Company", "Self-Insured Employer", "Group Self-Insurer", "State Insurance Fund"], help="Type of insurance provider.")

    # Collect all inputs into a dictionary
    inputs = {
        'accident_date': accident_date,
        'injury_date': injury_date,
        'age_at_injury': age_at_injury,
        'alternative_dispute_resolution': alternative_dispute_resolution,
        'assembly_date': assembly_date,
        'attorney_representative': attorney_representative,
        'average_weekly_wage': average_weekly_wage,
        'birth_year': birth_year,
        'c2_date': c2_date,
        'c3_date': c3_date,
        'carrier_name': carrier_name,
        'carrier_type': carrier_type,
        'claim_identifier': claim_identifier,
        'county_of_injury': county_of_injury,
        'covid_19_indicator': covid_19_indicator,
        'district_name': district_name,
        'first_hearing_date': first_hearing_date,
        'gender': gender,
        'ime4_count': ime4_count,
        'industry_code': industry_code,
        'industry_code_description': industry_code_description,
        'medical_fee_region': medical_fee_region,
        'oiics_nature_of_injury_description': oiics_nature_of_injury_description,
        'wcio_cause_of_injury_code': wcio_cause_of_injury_code,
        'wcio_cause_of_injury_description': wcio_cause_of_injury_description,
        'wcio_nature_of_injury_code': wcio_nature_of_injury_code,
        'wcio_nature_of_injury_description': wcio_nature_of_injury_description,
        'wcio_part_of_body_code': wcio_part_of_body_code,
        'wcio_part_of_body_description': wcio_part_of_body_description,
        'zip_code': zip_code,
        'agreement_reached': agreement_reached,
        'wcb_decision': wcb_decision
    }

    # A final call to action at the bottom
    st.markdown("**Please review all the information carefully before submitting.**")
    if st.button("Submit"):
        st.session_state.inputs = inputs
        st.session_state.page = "output"
        st.success("Your data has been submitted! You will be redirected to the output page.")





