import streamlit as st

def welcome_page():
    st.title("Welcome to the Claim Injury Type Prediction App")
    
    # Display the image
    image_url = "https://media.licdn.com/dms/image/v2/D4D3DAQFGx0XnuUvugA/image-scale_191_1128/image-scale_191_1128/0/1662458005755/nova_ims_information_management_school_cover?e=2147483647&v=beta&t=J3Q4LlZi36_4UAFhj2019QdtfXLn0kQwaX25jgaBhOQ"
    st.image(image_url, use_column_width=True)
    
    st.write("""
        This application predicts the **Claim Injury Type** for claims processed by the New York Workers' Compensation Board (WCB).
        
        **Instructions:**
        - Click the **Proceed** button to input claim-related data.
        - Fill out the form with accurate information.
        - Submit to receive a prediction of the injury type.
    """)
    if st.button("Proceed"):
        st.session_state.page = "input"



