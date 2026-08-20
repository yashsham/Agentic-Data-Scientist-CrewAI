# utils/llm_manager.py

import os
import logging
from crewai import LLM

# Disable CrewAI telemetry and 20s interactive trace prompt stall in non-interactive environments
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Fallback candidate models per provider
PROVIDER_MODELS = {
    "gemini": [
        os.getenv("GEMINI_MODEL", "gemini/gemini-2.0-flash"),
        "gemini/gemini-1.5-flash"
    ],
    "groq": [
        os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile"),
        "groq/llama3-70b-8192",
        "groq/llama-3.1-8b-instant"
    ],
    "nvidia": [
        os.getenv("NVIDIA_MODEL", "openai/meta/llama-3.3-70b-instruct"),
        "openai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "openai/nvidia/llama-3.1-nemotron-70b-instruct"
    ]
}


def get_secret(key_name: str):
    """Retrieves secret key from os.environ or streamlit secrets if available."""
    val = os.getenv(key_name)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return None


def build_llm_instance(provider: str, model_name: str = None) -> LLM:
    """
    Creates a CrewAI LLM instance based on provider and backend secrets.
    """
    provider = provider.lower()
    if not model_name:
        model_name = PROVIDER_MODELS.get(provider, [""])[0]

    kwargs = {"model": model_name, "temperature": 0.7}

    if provider == "gemini":
        key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        if key:
            kwargs["api_key"] = key
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key
    elif provider == "groq":
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

    return LLM(**kwargs)


def get_automatic_fallback_chain() -> list:
    """
    Automatically detects available API keys from environment/secrets
    and builds an ordered fallback list of (provider, model_name, LLM_instance).
    Priority: Gemini -> Groq -> NVIDIA (including model variations per provider)
    """
    chain = []

    # 1. Gemini Candidates
    if get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY"):
        for m in PROVIDER_MODELS["gemini"]:
            try:
                chain.append(("gemini", m, build_llm_instance("gemini", m)))
            except Exception as e:
                logger.warning(f"Could not build Gemini LLM ({m}): {e}")

    # 2. Groq Candidates
    if get_secret("GROQ_API_KEY"):
        for m in PROVIDER_MODELS["groq"]:
            try:
                chain.append(("groq", m, build_llm_instance("groq", m)))
            except Exception as e:
                logger.warning(f"Could not build Groq LLM ({m}): {e}")

    # 3. NVIDIA Candidates
    if get_secret("NVIDIA_API_KEY"):
        for m in PROVIDER_MODELS["nvidia"]:
            try:
                chain.append(("nvidia", m, build_llm_instance("nvidia", m)))
            except Exception as e:
                logger.warning(f"Could not build NVIDIA LLM ({m}): {e}")

    # Fallback default if no key explicitly matched
    if not chain:
        m = PROVIDER_MODELS["gemini"][0]
        chain.append(("gemini", m, build_llm_instance("gemini", m)))

    return chain
