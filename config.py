import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
    OPENROUTER_LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL","nvidia/nemotron-nano-9b-v2:free")

settings = Settings()