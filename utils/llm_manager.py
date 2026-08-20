# utils/llm_manager.py

import os
import logging
from crewai import LLM

# Disable CrewAI telemetry and interactive trace prompt stalls
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Updated provider models with verified active model names
PROVIDER_MODELS = {
    "groq": [
        os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-120b"),
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.6-27b",
        "groq/compound"
    ],
    "nvidia": [
        os.getenv("NVIDIA_MODEL", "openai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
        "openai/meta/llama-3.3-70b-instruct"
    ],
    "gemini": [
        os.getenv("GEMINI_MODEL", "gemini/gemini-2.0-flash"),
        "gemini/gemini-1.5-flash"
    ]
}


def sync_streamlit_secrets_to_env():
    """
    Syncs Streamlit st.secrets to os.environ for backend access across
    CrewAI, LiteLLM, and LangChain regardless of key casing.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            try:
                for k, v in st.secrets.items():
                    if isinstance(v, str):
                        val = v.strip()
                        os.environ[k] = val
                        os.environ[k.upper()] = val
                        os.environ[k.lower()] = val
                        if k.upper() in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
                            os.environ["GEMINI_API_KEY"] = val
                            os.environ["GOOGLE_API_KEY"] = val
            except Exception:
                pass
    except Exception:
        pass


def get_secret(key_name: str):
    """Retrieves secret key from os.environ or streamlit secrets if available."""
    sync_streamlit_secrets_to_env()
    val = os.getenv(key_name) or os.getenv(key_name.upper()) or os.getenv(key_name.lower())
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            try:
                if key_name in st.secrets:
                    return str(st.secrets[key_name]).strip()
                if key_name.upper() in st.secrets:
                    return str(st.secrets[key_name.upper()]).strip()
                if key_name.lower() in st.secrets:
                    return str(st.secrets[key_name.lower()]).strip()
            except Exception:
                pass
    except Exception:
        pass
    return None


def build_llm_instance(provider: str, model_name: str = None) -> LLM:
    """
    Creates a CrewAI LLM instance tailored for each provider with working stop parameters.
    """
    sync_streamlit_secrets_to_env()
    provider = provider.lower()
    if not model_name:
        model_name = PROVIDER_MODELS.get(provider, [""])[0]

    kwargs = {"model": model_name, "temperature": 0.7}

    if provider == "groq":
        key = get_secret("GROQ_API_KEY")
        if key:
            kwargs["api_key"] = key
            os.environ["GROQ_API_KEY"] = key

    elif provider == "nvidia":
        key = get_secret("NVIDIA_API_KEY")
        if key:
            kwargs["api_key"] = key
            os.environ["NVIDIA_API_KEY"] = key
        
        if not model_name.startswith("openai/"):
            kwargs["model"] = f"openai/{model_name}"
            
        kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"
        # Explicit non-empty stop sequence required by NVIDIA NIM API endpoint
        kwargs["stop"] = ["\n\nUser:"]

    elif provider == "gemini":
        key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        if key:
            kwargs["api_key"] = key
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key

    return LLM(**kwargs)


def get_automatic_fallback_chain() -> list:
    """
    Automatically detects available API keys from environment/secrets
    and builds an ordered fallback list of (provider, model_name, LLM_instance).
    Priority Order: Groq -> NVIDIA -> Gemini
    """
    sync_streamlit_secrets_to_env()
    chain = []

    # 1. Groq Candidates (Primary)
    if get_secret("GROQ_API_KEY"):
        for m in PROVIDER_MODELS["groq"]:
            try:
                chain.append(("groq", m, build_llm_instance("groq", m)))
            except Exception as e:
                logger.warning(f"Could not build Groq LLM ({m}): {e}")

    # 2. NVIDIA Candidates (First Fallback)
    if get_secret("NVIDIA_API_KEY"):
        for m in PROVIDER_MODELS["nvidia"]:
            try:
                chain.append(("nvidia", m, build_llm_instance("nvidia", m)))
            except Exception as e:
                logger.warning(f"Could not build NVIDIA LLM ({m}): {e}")

    # 3. Gemini Candidates (Second Fallback)
    if get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY"):
        for m in PROVIDER_MODELS["gemini"]:
            try:
                chain.append(("gemini", m, build_llm_instance("gemini", m)))
            except Exception as e:
                logger.warning(f"Could not build Gemini LLM ({m}): {e}")

    # Fallback default if no key explicitly matched
    if not chain:
        m = PROVIDER_MODELS["groq"][0]
        chain.append(("groq", m, build_llm_instance("groq", m)))

    return chain
