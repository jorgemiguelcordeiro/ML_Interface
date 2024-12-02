import streamlit as st

def toggle_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Light'
    theme = st.radio("Select Theme:", ["Light", "Dark"], key='theme_toggle')
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.experimental_rerun()

    if st.session_state.theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp {
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
            .stApp {
                background-color: #FFFFFF !important;
                color: #000000 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
