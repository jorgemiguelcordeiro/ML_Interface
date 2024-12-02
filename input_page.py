import streamlit as st

def input_page():
    st.title("Input Claim Data")

    # Example input fields (replace with your actual features)
    age = st.slider("Age", 18, 70, 30)
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract"])
    injury_severity = st.radio("Injury Severity", ["Minor", "Moderate", "Severe"])
    claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=1000.0)
    accident_description = st.text_area("Accident Description")

    # Collect inputs into a dictionary
    inputs = {
        'age': age,
        'employment_type': employment_type,
        'injury_severity': injury_severity,
        'claim_amount': claim_amount,
        'accident_description': accident_description
    }

    if st.button("Submit"):
        st.session_state.inputs = inputs
        st.session_state.page = "output"



