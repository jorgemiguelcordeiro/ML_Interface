import streamlit as st
#import streamlit_authenticator as stauth

st.set_page_config(page_title="Claim Injury Type Prediction App", layout="wide")

# Import page functions
from welcome_page import welcome_page
from input_page import input_page
from output_page import output_page


    
# Include theme toggle
#toggle_theme()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Page Navigation
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'input':
    input_page()
elif st.session_state.page == 'output':
    output_page()



