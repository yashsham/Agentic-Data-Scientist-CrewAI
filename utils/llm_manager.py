# utils/llm_manager.py

import os
import logging
from crewai import LLM

logger = logging.getLogger(__name__)

# Default model definitions for supported providers
DEFAULT_MODELS = {
    "gemini": "gemini/gemini-2.0-flash",
    "groq": "groq/llama-3.3-70b-versatile",
    "nvidia": "openai/meta/llama-3.3-70b-instruct"
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
        model_name = DEFAULT_MODELS.get(provider, "")

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
        kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"

    return LLM(**kwargs)


def get_automatic_fallback_chain() -> list:
    """
    Automatically detects available API keys from environment/secrets
    and builds an ordered fallback list of (provider, model_name, LLM_instance).
    Default priority: Gemini -> Groq -> NVIDIA
    """
    chain = []

    # Check Gemini
    if get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY"):
        try:
            m = DEFAULT_MODELS["gemini"]
            chain.append(("gemini", m, build_llm_instance("gemini", m)))
        except Exception as e:
            logger.warning(f"Could not build Gemini LLM: {e}")

    # Check Groq
    if get_secret("GROQ_API_KEY"):
        try:
            m = DEFAULT_MODELS["groq"]
            chain.append(("groq", m, build_llm_instance("groq", m)))
        except Exception as e:
            logger.warning(f"Could not build Groq LLM: {e}")

    # Check NVIDIA
    if get_secret("NVIDIA_API_KEY"):
        try:
            m = DEFAULT_MODELS["nvidia"]
            chain.append(("nvidia", m, build_llm_instance("nvidia", m)))
        except Exception as e:
            logger.warning(f"Could not build NVIDIA LLM: {e}")

    # Fallback default if no key explicitly matched
    if not chain:
        m = DEFAULT_MODELS["gemini"]
        chain.append(("gemini", m, build_llm_instance("gemini", m)))

    return chain
