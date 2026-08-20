# app.py

import streamlit as st
import os
import glob
import pandas as pd
from main import run_crew  # Import the function from main.py

# --- Page Configuration ---
st.set_page_config(page_title="Agentic Data Scientist Crew", page_icon="🤖", layout="wide")
st.title("🤖 Agentic Data Scientist Crew")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Trigger button for running full analysis
    run_button = st.button("🚀 Run Analysis Crew", disabled=(uploaded_file is None), use_container_width=True)

# --- Main Area ---
if run_button and uploaded_file:
    # 1. Save the uploaded file
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"File '{uploaded_file.name}' saved. Kicking off the agent crew with backend failover support...")

    # Status toast helper
    def status_toast(event):
        status = event.get("status")
        msg = event.get("message", "")
        if status == "fallback":
            st.toast(msg, icon="🔄")
        elif status == "failed":
            st.toast(f"Provider failed: {event.get('provider')}", icon="⚠️")

    # 2. Run the agent crew with automatic backend LLM fallback (Gemini -> Groq -> NVIDIA)
    with st.spinner("The AI crew is analyzing your data... please wait."):
        try:
            result = run_crew(save_path, status_callback=status_toast)
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Execution Error: {e}")
            result = None

    # 3. Display the final report
    st.header("Final Analysis Report")
    report_path = os.path.join("reports", "final_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.error("Report file not found.")
        if result:
            st.write(f"Crew's final output: {result}")

    # 4. Find and display the generated plots
    st.header("Generated Visualizations")
    image_files = glob.glob("reports/*.png")
    if image_files:
        cols = st.columns(min(len(image_files), 2))
        for idx, image_file in enumerate(image_files):
            col = cols[idx % len(cols)]
            with col:
                st.image(image_file, caption=os.path.basename(image_file), use_container_width=True)
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