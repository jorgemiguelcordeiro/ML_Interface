pip install streamlit-lottie


import streamlit as st
import os
import base64
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def welcome_page():
    # Define the image file name
    image_university = 'image_university.jpg'  # Ensure the correct file name and extension

    # Get the absolute path to the directory containing this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the image file
    image_path = os.path.join(BASE_DIR, image_university)

    # Read and encode the image
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        encoded_image = base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error("Background image not found.")
        encoded_image = None
    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")
        encoded_image = None

    # Inject CSS only if the image was loaded successfully
    if encoded_image:
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        h1, h2, h3, h4, h5, h6, p, div, span {{
            color: white !important;
        }}
        .stButton {{
            background-color: rgba(0, 0, 0, 0.5);
            padding: 5px;
            border-radius: 10px;
            width: 200px; /* Adjust as needed */
            margin: 0 auto; /* Center the container */
        }}
        .stButton>button {{
            color: white !important;
            background-color: #1E90FF !important;
            border-radius: 10px;
            width: 100%;
            height: 50px;
            font-size: 18px;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        st.write("Using default background.")

    st.title("Welcome to the Claim Injury Type Prediction App")

    st.write("""
        This application predicts the **Claim Injury Type** for claims processed by the New York Workers' Compensation Board (WCB).
        
        **Instructions:**
        - Click the **Proceed** button to begin.
        - Fill out the form with accurate information.
        - Submit to receive a prediction of the injury type.
    """)

    # Check if the user already clicked "Proceed"
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = False

    if not st.session_state.show_animation:
        # User has not clicked proceed yet
        if st.button("Proceed"):
            # Show animation next time around
            st.session_state.show_animation = True
            st.experimental_rerun()
    else:
        # User clicked proceed: show animation
        st.write("Loading the next page... Please wait for a moment.")
        
        # Load and display a Lottie animation (replace the URL with your chosen animation)
        # Example Lottie file URL (this one is just a placeholder)
        lottie_url = "<iframe src="https://giphy.com/embed/uIJBFZoOaifHf52MER" width="480" height="446" style="" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/UniversalMusicIndia-elvish-dg-immortals-bawli-uIJBFZoOaifHf52MER">via GIPHY</a></p>"
        lottie_animation = load_lottieurl(lottie_url)
        
        if lottie_animation:
            st_lottie(lottie_animation, height=300)

        st.write("**Done!** Ready to proceed to the input page?")
        if st.button("Continue to Input Page"):
            st.session_state.page = "input"
            st.experimental_rerun()

    # Add a horizontal line to separate the content from the footer
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # Add the disclaimer at the bottom
    st.markdown("""
    <div style="text-align:center; font-size:12px; color:gray;">
    This application was developed as part of a university project. All rights are reserved to the students involved in its creation. The data and results are intended for educational purposes only.
    </div>
    """, unsafe_allow_html=True)



