import streamlit as st
import os
import datetime

from welcome_page import welcome_page
from input_page import input_page
from output_page import output_page
#from eda_page import eda_page

st.set_page_config(page_title="Claim Injury Type Prediction App", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Sidebar Navigation
#st.sidebar.title("Navigation")
#page_selection = st.sidebar.radio("Go to", ('Welcome', 'EDA', 'Prediction'))

# Map sidebar selection to st.session_state.page
if page_selection == 'Welcome':
    st.session_state.page = 'welcome'
#elif page_selection == 'EDA':
    #st.session_state.page = 'eda'
elif page_selection == 'Prediction':
    # If you still want the 3-step approach: Welcome -> Input -> Output
    # Consider directly jumping to input or keep it as is.
    # For simplicity, let's jump directly to input:
    st.session_state.page = 'input'

# Page Navigation
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'input':
    input_page()
elif st.session_state.page == 'output':
    output_page()
#elif st.session_state.page == 'eda':
    #eda_page()



