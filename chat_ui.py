from pathlib import Path
import streamlit as st

CSS_PATH = Path(__file__).parent / "static" / "chat_ui.css"

def inject_chat_ui_css():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_chat_message(role: str, content:str):
    css_class = "user" if role == "user" else "assistant"

    safe_content = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )

    st.markdown(
        f'<div class="chat-row {css_class}">'
        f'<div class="chat-bubble {css_class}">{safe_content}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )