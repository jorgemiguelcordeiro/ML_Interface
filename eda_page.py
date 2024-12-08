pip install matplotlib


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def eda_page():
    st.title("Dataset Overview")

    # Load your data
    # Adjust the path as needed
    data_path = "train_data/train_data"
    df = pd.read_csv(data_path)

    st.write("Below is a sample of the dataset:")
    st.dataframe(df.head(10))  # Show first 10 rows

    # Add filters: for example, filter by date range if 'accident_date' column exists
    if 'accident_date' in df.columns:
        st.sidebar.markdown("### Filter by Date Range")
        start_date = st.sidebar.date_input("Start Date", value=df['accident_date'].min())
        end_date = st.sidebar.date_input("End Date", value=df['accident_date'].max())

        # Ensure the date columns are in datetime format
        if df['accident_date'].dtype == 'object':
            df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')

        filtered_df = df[(df['accident_date'] >= pd.to_datetime(start_date)) &
                         (df['accident_date'] <= pd.to_datetime(end_date))]
    else:
        filtered_df = df

    st.write(f"**Number of records after filtering**: {len(filtered_df)}")

    # Add a simple plot: e.g., distribution of a numeric column 'age_at_injury'
    if 'age_at_injury' in filtered_df.columns:
        st.write("### Graphs and Visualizations")
        st.write("Distribution of Age at Injury")
        
        fig, ax = plt.subplots()
        filtered_df['age_at_injury'].hist(ax=ax, bins=20)
        ax.set_xlabel("Age at Injury")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Age at Injury")
        st.pyplot(fig)
    else:
        st.write("No 'age_at_injury' column found for plotting.")

    # Add a button to proceed to input page
    if st.button("Proceed to Prediction"):
        st.session_state.page = "input"

