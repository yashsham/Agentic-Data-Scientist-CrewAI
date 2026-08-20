# app.py

import streamlit as st
import os
import glob
import pandas as pd
from utils.llm_manager import sync_streamlit_secrets_to_env
from main import run_crew

# Automatically sync Streamlit Secrets to environment variables
sync_streamlit_secrets_to_env()

# --- Page Configuration ---
st.set_page_config(page_title="Agentic Data Scientist Crew", page_icon="🤖", layout="wide")
st.title("🤖 Agentic Data Scientist Crew")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    run_button = st.button("🚀 Run Analysis Crew", disabled=(uploaded_file is None), use_container_width=True)

# --- Main Area ---
if run_button and uploaded_file:
    # 1. Save uploaded dataset
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"File '{uploaded_file.name}' saved. Starting Agentic Crew (Backend Fallback: Groq ➔ NVIDIA ➔ Gemini)...")

    status_history = []
    log_expander = st.expander("🔍 Live Agent & Model Fallback Logs", expanded=True)
    log_area = log_expander.empty()

    def status_callback(event):
        st_type = event.get("status")
        msg = event.get("message", "")
        if st_type == "attempting":
            status_history.append(f"⏳ {msg}")
            st.toast(msg, icon="⏳")
        elif st_type == "fallback":
            status_history.append(f"🔄 {msg}")
            st.toast(msg, icon="🔄")
        elif st_type == "failed":
            status_history.append(f"❌ {msg} (Error: {event.get('error')})")
            st.toast(f"Provider failed: {event.get('provider')}", icon="⚠️")
        elif st_type == "success":
            status_history.append(f"✅ {msg}")
            st.toast(msg, icon="🎉")
        
        log_area.code("\n".join(status_history), language="text")

    # 2. Run agent crew with automatic backend failover (Groq -> NVIDIA -> Gemini)
    with st.spinner("The AI crew is analyzing your data... please wait."):
        try:
            result = run_crew(save_path, status_callback=status_callback)
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.warning("Tip: Make sure at least one valid API key (`GROQ_API_KEY`, `NVIDIA_API_KEY`, or `GEMINI_API_KEY`) is set in your Streamlit Secrets.")
            result = None

    # 3. Display final report
    st.header("Final Analysis Report")
    report_path = os.path.join("reports", "final_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.error("Report file not found.")
        if result:
            st.write(f"Crew's final output: {result}")

    # 4. Display generated visualizations
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
    # Initial view or dataset preview
    st.header("Analysis Report")
    if uploaded_file:
        st.subheader("Data Preview")
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
    else:
        st.info("Upload a CSV file and click 'Run Analysis Crew' to see the results.")