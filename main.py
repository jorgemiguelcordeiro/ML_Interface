
import streamlit as st

# Import page functions
from welcome_page import welcome_page
from input_page import input_page
from output_page import output_page

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

# Include theme toggle
toggle_theme()
