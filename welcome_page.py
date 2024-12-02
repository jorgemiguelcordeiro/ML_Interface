import streamlit as st

def welcome_page():
    st.title("Welcome to the Claim Injury Type Prediction App")
    st.write("""
        This application predicts the **Claim Injury Type** for claims processed by the New York Workers' Compensation Board (WCB).
        
        **Instructions:**
        - Click the **Proceed** button to input claim-related data.
        - Fill out the form with accurate information.
        - Submit to receive a prediction of the injury type.
    """)
    if st.button("Proceed"):
        st.session_state.page = "input"



