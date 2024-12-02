# toggle_theme.py

import streamlit as st

def toggle_theme():
    # Initialize the theme in session state if it doesn't exist
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Light'

    # Radio button to select theme
    theme = st.radio("Select Theme:", ["Light", "Dark"], key='theme_toggle')

    # Update the theme in session state
    st.session_state.theme = theme

    # Apply the CSS styles based on the selected theme
    if st.session_state.theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0E1117;
                color: #FFFFFF;
            }
            /* Additional CSS styles for dark theme */
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
            /* Additional CSS styles for light theme */
            </style>
            """,
            unsafe_allow_html=True
        )
