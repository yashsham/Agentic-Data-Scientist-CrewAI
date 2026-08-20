# app.py

import streamlit as st
import os
import glob
import pandas as pd
from dotenv import load_dotenv
from main import run_crew
from utils.llm_manager import AVAILABLE_MODELS, DEFAULT_MODELS, build_llm_instance

# Load initial .env environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic Data Scientist Crew",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .status-badge-active {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-inactive {
        background-color: #F3F4F6;
        color: #6B7280;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🤖 Agentic Data Scientist Crew</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated End-to-End Data Analysis with Multi-LLM Fallback (Gemini, Groq & NVIDIA)</div>', unsafe_allow_html=True)

# --- Sidebar: API Keys & Model Setup ---
with st.sidebar:
    st.header("🔑 LLM API Configuration")
    st.caption("Provide API keys for providers to enable automatic failover.")

    # 1. Gemini Config
    gemini_key_env = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        value=gemini_key_env,
        type="password",
        help="Required for Gemini models. Sets GEMINI_API_KEY / GOOGLE_API_KEY."
    )
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        st.markdown('Status: <span class="status-badge-active">🟢 Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('Status: <span class="status-badge-inactive">⚪ Not set</span>', unsafe_allow_html=True)

    st.divider()

    # 2. Groq Config
    groq_key_env = os.getenv("GROQ_API_KEY") or ""
    groq_api_key = st.text_input(
        "Groq API Key",
        value=groq_key_env,
        type="password",
        help="Required for Groq models (Llama 3.3, Mixtral, etc.). Sets GROQ_API_KEY."
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.markdown('Status: <span class="status-badge-active">🟢 Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('Status: <span class="status-badge-inactive">⚪ Not set</span>', unsafe_allow_html=True)

    st.divider()

    # 3. NVIDIA Config
    nvidia_key_env = os.getenv("NVIDIA_API_KEY") or ""
    nvidia_api_key = st.text_input(
        "NVIDIA NIM API Key",
        value=nvidia_key_env,
        type="password",
        help="Required for NVIDIA AI Endpoints. Sets NVIDIA_API_KEY."
    )
    if nvidia_api_key:
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
        st.markdown('Status: <span class="status-badge-active">🟢 Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('Status: <span class="status-badge-inactive">⚪ Not set</span>', unsafe_allow_html=True)

    st.divider()
    st.header("⚙️ Fallback Chain Settings")

    # Select Primary Provider
    all_providers = ["gemini", "groq", "nvidia"]
    primary_provider = st.selectbox(
        "Primary Provider",
        options=all_providers,
        index=0,
        help="The first LLM provider to attempt."
    )

    remaining_providers = [p for p in all_providers if p != primary_provider]
    fallback_1 = st.selectbox(
        "First Fallback Provider",
        options=["None"] + remaining_providers,
        index=1 if len(remaining_providers) > 0 else 0
    )

    fallback_2_options = [p for p in remaining_providers if p != fallback_1]
    fallback_2 = st.selectbox(
        "Second Fallback Provider",
        options=["None"] + fallback_2_options,
        index=1 if len(fallback_2_options) > 0 else 0
    )

    # Specific Model Selection per Provider
    st.markdown("##### Specific Model Selection")
    gemini_model = st.selectbox("Gemini Model", AVAILABLE_MODELS["gemini"], index=0)
    groq_model = st.selectbox("Groq Model", AVAILABLE_MODELS["groq"], index=0)
    nvidia_model = st.selectbox("NVIDIA Model", AVAILABLE_MODELS["nvidia"], index=0)

    model_map = {
        "gemini": gemini_model,
        "groq": groq_model,
        "nvidia": nvidia_model
    }

    key_map = {
        "gemini": gemini_api_key,
        "groq": groq_api_key,
        "nvidia": nvidia_api_key
    }

# --- Main Layout ---
col_upload, col_preview = st.columns([1, 2])

with col_upload:
    st.subheader("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: **{uploaded_file.name}**")

    # Construct configured LLM chain
    user_chain = []
    selected_providers = [primary_provider]
    if fallback_1 != "None":
        selected_providers.append(fallback_1)
    if fallback_2 != "None":
        selected_providers.append(fallback_2)

    for prov in selected_providers:
        m = model_map[prov]
        k = key_map[prov]
        try:
            llm_inst = build_llm_instance(prov, model_name=m, api_key=k)
            user_chain.append((prov, m, llm_inst))
        except Exception as e:
            st.warning(f"Could not prepare {prov.upper()} model: {e}")

    run_disabled = (uploaded_file is None) or (len(user_chain) == 0)
    
    if len(user_chain) == 0 and uploaded_file is not None:
        st.error("Please enter at least one API key in the sidebar!")

    run_button = st.button("🚀 Run Analysis Crew", disabled=run_disabled, use_container_width=True, type="primary")

with col_preview:
    st.subheader("2. Data Preview")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            tab_head, tab_info, tab_stats = st.tabs(["Dataset Head", "Data Types & Nulls", "Summary Statistics"])
            with tab_head:
                st.dataframe(df.head(10), use_container_width=True)
            with tab_info:
                info_df = pd.DataFrame({
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str),
                    "Null Count": df.isnull().sum(),
                    "Null %": (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(info_df, use_container_width=True)
            with tab_stats:
                st.dataframe(df.describe(), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        st.info("Upload a CSV file on the left to preview dataset details.")

st.divider()

# --- Execution & Results Section ---
if run_button and uploaded_file:
    st.subheader("3. Execution Progress & Fallback Log")
    
    log_container = st.empty()
    logs_history = []

    def update_status(event):
        status_type = event.get("status")
        msg = event.get("message", "")
        if status_type == "attempting":
            logs_history.append(f"ℹ️ {msg}")
            st.toast(msg, icon="⏳")
        elif status_type == "success":
            logs_history.append(f"✅ {msg}")
            st.toast(msg, icon="🎉")
        elif status_type == "failed":
            logs_history.append(f"❌ {msg}")
            st.toast(f"Failed attempt: {event.get('provider')}", icon="⚠️")
        elif status_type == "fallback":
            logs_history.append(f"🔄 {msg}")
            st.toast(msg, icon="🔄")
            
        log_container.code("\n".join(logs_history), language="text")

    save_path = os.path.join("data", uploaded_file.name)
    
    with st.spinner("AI Agents are executing (Fetching -> Cleaning -> Visualizing -> Reporting)..."):
        try:
            exec_info = run_crew(
                filepath=save_path,
                llm_chain=user_chain,
                status_callback=update_status
            )
            st.success(f"Analysis Complete! Executed successfully using **{exec_info['provider'].upper()}** (`{exec_info['model']}`).")
            if exec_info.get("used_fallback"):
                st.info(f"Fallback mechanism was triggered! Took {exec_info['attempts']} attempt(s) to succeed.")
        except Exception as err:
            st.error(f"Execution Error: {err}")

# --- Display Final Report & Visualizations ---
st.subheader("4. Analysis Output & Visualizations")

tab_report, tab_plots = st.tabs(["📄 Final Report", "📊 Generated Visualizations"])

with tab_report:
    report_path = os.path.join("reports", "final_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        
        st.download_button(
            label="📥 Download Report (.md)",
            data=report_text,
            file_name="final_report.md",
            mime="text/markdown"
        )
        st.markdown(report_text)
    else:
        st.info("No report generated yet. Run the analysis crew to create a report.")

with tab_plots:
    image_files = glob.glob("reports/*.png")
    if image_files:
        st.write(f"Found **{len(image_files)}** visualization(s):")
        cols = st.columns(min(len(image_files), 2))
        for idx, img_path in enumerate(image_files):
            col_target = cols[idx % len(cols)]
            with col_target:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                with open(img_path, "rb") as file:
                    st.download_button(
                        label=f"📥 Download {os.path.basename(img_path)}",
                        data=file,
                        file_name=os.path.basename(img_path),
                        mime="image/png",
                        key=f"dl_btn_{idx}"
                    )
    else:
        st.info("No visualizations found in the 'reports' directory.")