import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from google import genai
from google.genai import types
from dotenv import load_dotenv


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")



@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    """
    Create a Gemini client. Prefer Streamlit secrets if present, else fall back to env var.
    This avoids StreamlitSecretNotFoundError when no secrets.toml exists.
    """
    api_key = None


    try:

        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
    except Exception:

        api_key = st.secrets["GEMINI_API_KEY"]

    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY either in "
            ".streamlit/secrets.toml or as an environment variable."
        )

    return genai.Client(api_key=api_key)



def upload_to_gemini(uploaded_file):
    """Upload a Streamlit UploadedFile to Gemini Files and return the File object."""
    if uploaded_file is None:
        return None

    client = get_client()

    suffix = Path(uploaded_file.name).suffix or ""
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name


    mime = getattr(uploaded_file, "type", None) or "application/octet-stream"

    file_obj = client.files.upload(
        file=temp_path,
        config=types.UploadFileConfig(
            mime_type=mime,
            display_name=uploaded_file.name,
        ),
    )
    return file_obj


def build_contents(history, file_obj=None):
    """
    Convert our simple [{role:'user'|'assistant', content:str}, ...] history
    into google-genai 'contents'. Optionally include an uploaded file.
    """
    contents = []


    if file_obj is not None:
        contents.append(file_obj)


    last_turns = history[-8:] if history else []
    for m in last_turns:
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=m["content"])]
            )
        )
    return contents



def chat_once(biosignal_label: str, history, user_prompt: str, file_obj=None) -> str:
    """
    Single-turn generation using stateless generate_content;
    we supply short history + (optional) file each call.
    """
    client = get_client()


    ext_history = (history or []) + [{"role": "user", "content": user_prompt}]

    contents = build_contents(ext_history, file_obj=file_obj)

    system_instruction = (
        f"You are a careful, helpful assistant specialized in {biosignal_label} biosignals. "
        "When a document is uploaded, analyze it and cite sections by page/section names when possible. "
        "Be explicit about assumptions and limitations. If the file looks like raw data, describe possible preprocessing "
        "(e.g., filtering, artifact removal) at a high level and answer questions about interpretations."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    return (response.text or "").strip()
