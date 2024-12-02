import streamlit as st

def welcome_page():
    
    # Display the image
    image_url = "https://media.licdn.com/dms/image/v2/D4D3DAQFGx0XnuUvugA/image-scale_191_1128/image-scale_191_1128/0/1662458005755/nova_ims_information_management_school_cover?e=2147483647&v=beta&t=J3Q4LlZi36_4UAFhj2019QdtfXLn0kQwaX25jgaBhOQ"
    st.image(image_url, use_column_width=True)

    # CSS for animations
    st.markdown("""
    <style>
    .fade-in-text {
        animation: fadeIn 3s;
    }

    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
    """, unsafe_allow_html=True)

    # Animated text
    st.markdown('<div class="fade-in-text"><p>This application predicts the <strong>Claim Injury Type</strong> for claims processed by the New York Workers\' Compensation Board (WCB).</p></div>', unsafe_allow_html=True)

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



