import streamlit as st
import os
import base64

def welcome_page():

    st.set_page_config(page_title="Claim Injury Type Prediction App", layout="wide")

    # Get the absolute path to the background image
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    background_image = os.path.join(BASE_DIR, 'image_university')  # Adjusted path

    # Read the image file and encode it to base64
    with open(image_university, 'rb') as f:
        data = f.read()
    encoded_image = base64.b64encode(data).decode()

    # Inject CSS with the background image and overlay
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: white !important;
    }}
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    
    # Display the image
    image_url = "https://media.licdn.com/dms/image/v2/D4D3DAQFGx0XnuUvugA/image-scale_191_1128/image-scale_191_1128/0/1662458005755/nova_ims_information_management_school_cover?e=2147483647&v=beta&t=J3Q4LlZi36_4UAFhj2019QdtfXLn0kQwaX25jgaBhOQ"
    st.image(image_url, use_column_width=True)

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

    # Add a horizontal line to separate the content from the footer
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # Add the disclaimer at the bottom
    st.markdown("""
    <div style="text-align:center; font-size:12px; color:gray;">
    This application was developed as part of a university project. All rights are reserved to the students involved in its creation. The data and results are intended for educational purposes only.
    </div>
    """, unsafe_allow_html=True)



