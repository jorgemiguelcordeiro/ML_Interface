
def toggle_theme():
    theme = st.radio("Select Theme:", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #0E1117;
                color: #FFFFFF;
            }
            </style>
            """,
            unsafe_allow_html=True
        )



