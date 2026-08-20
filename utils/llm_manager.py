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

AVAILABLE_MODELS = {
    "gemini": [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
    ],
    "groq": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/mixtral-8x7b-32768",
    ],
    "nvidia": [
        "openai/meta/llama-3.3-70b-instruct",
        "openai/nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia_ai_endpoints/meta/llama-3.3-70b-instruct",
    ]
}


def build_llm_instance(provider: str, model_name: str = None, api_key: str = None) -> LLM:
    """
    Creates a CrewAI LLM instance based on provider, model_name, and API key.
    """
    provider = provider.lower()
    if not model_name:
        model_name = DEFAULT_MODELS.get(provider, "")

    kwargs = {"model": model_name, "temperature": 0.7}

    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            kwargs["api_key"] = key
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key
    elif provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if key:
            kwargs["api_key"] = key
            os.environ["GROQ_API_KEY"] = key
    elif provider == "nvidia":
        key = api_key or os.getenv("NVIDIA_API_KEY")
        if key:
            kwargs["api_key"] = key
            os.environ["NVIDIA_API_KEY"] = key
        # NVIDIA API endpoint setting
        kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"

    return LLM(**kwargs)


def get_configured_llm_chain(user_config: list = None) -> list:
    """
    Returns an ordered list of tuples: (provider, model_name, LLM_instance) based on available keys or user preferences.
    user_config: list of dicts [{'provider': 'groq', 'model': '...', 'api_key': '...'}, ...]
    """
    llm_chain = []

    if user_config:
        for cfg in user_config:
            prov = cfg.get("provider")
            model = cfg.get("model") or DEFAULT_MODELS.get(prov)
            key = cfg.get("api_key")
            
            env_key_names = {
                "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                "groq": ["GROQ_API_KEY"],
                "nvidia": ["NVIDIA_API_KEY"]
            }
            has_key = bool(key) or any(os.getenv(k) for k in env_key_names.get(prov, []))
            
            if has_key:
                try:
                    llm_obj = build_llm_instance(prov, model, key)
                    llm_chain.append((prov, model, llm_obj))
                except Exception as e:
                    logger.warning(f"Could not build LLM for provider {prov}: {e}")

    # Fallback to default check if no user_config given or chain is empty
    if not llm_chain:
        for prov in ["gemini", "groq", "nvidia"]:
            env_key_names = {
                "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                "groq": ["GROQ_API_KEY"],
                "nvidia": ["NVIDIA_API_KEY"]
            }
            if any(os.getenv(k) for k in env_key_names.get(prov, [])):
                try:
                    model = DEFAULT_MODELS[prov]
                    llm_obj = build_llm_instance(prov, model)
                    llm_chain.append((prov, model, llm_obj))
                except Exception as e:
                    logger.warning(f"Error instantiating default {prov} LLM: {e}")

    return llm_chain
