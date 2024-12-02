
import streamlit as st

def toggle_theme():
    theme = st.radio("Select Theme:", ["Light", "Dark"], key='theme_toggle')
    if theme == "Dark":
        st.markdown(
            """
            <style>
            html, body, [class*="css"]  {
                background-color: #0E1117 !important;
                color: #FFFFFF !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            html, body, [class*="css"]  {
                background-color: #FFFFFF !important;
                color: #000000 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
