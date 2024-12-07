import streamlit as st
import datetime

def input_page():
    st.title("Input Claim Data")

    # 1. Accident Date
    accident_date = st.date_input("Accident Date", value=datetime.date.today())

    # 2. Injury Date of the Claim (Assuming it's the same as Accident Date)
    injury_date = st.date_input("Injury Date", value=datetime.date.today())

    # 3. Age at Injury
    age_at_injury = st.number_input("Age at Injury", min_value=0, max_value=100, value=30)

    # 4. Alternative Dispute Resolution
    alternative_dispute_resolution = st.selectbox("Alternative Dispute Resolution", ["Yes", "No"])

    # 5. Assembly Date
    assembly_date = st.date_input("Assembly Date", value=datetime.date.today())

    # 6. Attorney/Representative
    attorney_representative = st.selectbox("Attorney/Representative", ["Yes", "No"])

    # 7. Average Weekly Wage
    average_weekly_wage = st.number_input("Average Weekly Wage ($)", min_value=0.0, value=1000.0)

    # 8. Birth Year
    birth_year = st.number_input("Birth Year", min_value=1900, max_value=datetime.date.today().year, value=1990)

    # 9. C-2 Date
    c2_date = st.date_input("C-2 Date", value=datetime.date.today())

    # 10. C-3 Date
    c3_date = st.date_input("C-3 Date", value=datetime.date.today())

    # 11. Carrier Name
    carrier_name = st.text_input("Carrier Name")

    # 12. Carrier Type
    carrier_type = st.selectbox("Carrier Type", ["Insurance Company", "Self-Insured Employer", "Group Self-Insurer", "State Insurance Fund"])

    # 13. Claim Identifier
    claim_identifier = st.text_input("Claim Identifier")

    # 14. County of Injury
    counties = ["Albany", "Bronx", "Brooklyn", "Queens", "Richmond", "Manhattan", "Suffolk", "Nassau", "Westchester"]  # Add all NY counties
    county_of_injury = st.selectbox("County of Injury", counties)

    # 15. COVID-19 Indicator
    covid_19_indicator = st.selectbox("COVID-19 Indicator", ["Yes", "No"])

    # 16. District Name
    districts = ["Albany", "Binghamton", "Brooklyn", "Buffalo", "Long Island", "Manhattan", "Queens", "Rochester", "Syracuse"]  # Example districts
    district_name = st.selectbox("District Name", districts)

    # 17. First Hearing Date
    first_hearing_date = st.date_input("First Hearing Date", value=datetime.date.today())

    # 18. Gender
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])

    # 19. IME-4 Count
    ime4_count = st.number_input("IME-4 Count", min_value=0, value=0)

    # 20. Industry Code
    industry_code = st.text_input("Industry Code")

    # 21. Industry Code Description
    industry_code_descriptions = ["Agriculture", "Construction", "Manufacturing", "Retail Trade", "Healthcare"]  # Example descriptions
    industry_code_description = st.selectbox("Industry Code Description", industry_code_descriptions)

    # 22. Medical Fee Region
    medical_fee_regions = ["Upstate", "Downstate", "NYC"]  # Example regions
    medical_fee_region = st.selectbox("Medical Fee Region", medical_fee_regions)

    # 23. OIICS Nature of Injury Description
    oiics_nature_descriptions = ["Fractures", "Sprains", "Cuts", "Burns"]  # Example descriptions
    oiics_nature_of_injury_description = st.selectbox("OIICS Nature of Injury Description", oiics_nature_descriptions)

    # 24. WCIO Cause of Injury Code
    wcio_cause_of_injury_code = st.text_input("WCIO Cause of Injury Code")

    # 25. WCIO Cause of Injury Description
    wcio_cause_descriptions = ["Fall", "Struck by object", "Motor vehicle accident"]  # Example descriptions
    wcio_cause_of_injury_description = st.selectbox("WCIO Cause of Injury Description", wcio_cause_descriptions)

    # 26. WCIO Nature of Injury Code
    wcio_nature_of_injury_code = st.text_input("WCIO Nature of Injury Code")

    # 27. WCIO Nature of Injury Description
    wcio_nature_descriptions = ["Fracture", "Laceration", "Burn"]  # Example descriptions
    wcio_nature_of_injury_description = st.selectbox("WCIO Nature of Injury Description", wcio_nature_descriptions)

    # 28. WCIO Part Of Body Code
    wcio_part_of_body_code = st.text_input("WCIO Part Of Body Code")

    # 29. WCIO Part Of Body Description
    wcio_part_of_body_descriptions = ["Head", "Back", "Arm", "Leg"]  # Example descriptions
    wcio_part_of_body_description = st.selectbox("WCIO Part Of Body Description", wcio_part_of_body_descriptions)

    # 30. Zip Code
    zip_code = st.text_input("Zip Code")

    # 31. Agreement Reached
    agreement_reached = st.selectbox("Agreement Reached", ["Yes", "No"])

    # 32. WCB Decision
    wcb_decision = st.selectbox("WCB Decision", ["Accident", "Occupational Disease", "Pending"])

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

    if st.button("Submit"):
        st.session_state.inputs = inputs
        st.session_state.page = "output"



