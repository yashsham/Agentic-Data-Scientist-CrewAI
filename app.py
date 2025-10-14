# app.py

import streamlit as st
import os
import pandas as pd
from main import run_crew # Import the function from main.py
import glob # Used to find the image files

# --- Page Configuration ---
st.set_page_config(page_title="Agentic Data Scientist", layout="wide")
st.title("🤖 Agentic Data Scientist Crew")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # This button will now trigger the full analysis
    run_button = st.button("🚀 Run Analysis Crew", disabled=(uploaded_file is None))

# --- Main Area ---
if run_button:
    # 1. Save the uploaded file
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"File '{uploaded_file.name}' saved. Kicking off the agent crew...")

    # 2. Run the agent crew
    with st.spinner("The AI crew is analyzing your data... please wait."):
        result = run_crew(save_path) # Call the function
        st.success("Analysis complete!")

    # 3. Display the final report
    st.header("Final Analysis Report")
    report_path = "reports/final_report.md"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            st.markdown(f.read())
    else:
        st.error("Report file not found.")
        st.write(f"Crew's final output: {result}")

    # 4. Find and display the generated plots
    st.header("Generated Visualizations")
    image_files = glob.glob("reports/*.png")
    if image_files:
        for image_file in image_files:
            st.image(image_file, caption=os.path.basename(image_file))
    else:
        st.warning("No image files were found in the 'reports' directory.")

else:
    # Initial view or after file upload
    st.header("Analysis Report")
    if uploaded_file:
        st.subheader("Data Preview")
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
    else:
        st.info("Upload a CSV file and click 'Run Analysis Crew' to see the results.")