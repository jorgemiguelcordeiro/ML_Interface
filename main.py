!pip install streamlit-authenticator

import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(page_title="Claim Injury Type Prediction App", layout="wide")
# Import page functions
from welcome_page import welcome_page
from input_page import input_page
from output_page import output_page
from toggle_theme import toggle_theme
    
# Include theme toggle
#toggle_theme()

# Define user credentials
names = ['John Doe', 'Jane Smith']
usernames = ['johndoe', 'janesmith']
passwords = ['password123', 'mysecurepassword']
hashed_passwords = stauth.Hasher(passwords).generate()

# Create authenticator
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

# Add login widget
name, authentication_status, username = authenticator.login('Login', 'main')

# Check authentication status
if authentication_status:
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'
        # Page navigation logic
    if st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'input':
        input_page()
    elif st.session_state.page == 'output':
        output_page()

elif authentication_status == False:
    st.error('Username or password is incorrect')

elif authentication_status == None:
    st.warning('Please enter your username and password')
'''
# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Page Navigation
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'input':
    input_page()
elif st.session_state.page == 'output':
    output_page()'''

