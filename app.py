import os
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file. or your cloud environment")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Biosignal & Imaging", layout="wide")
st.title("Biosignal & Imaging")

MODALITIES = ["VMG", "CT", "Speech", "ECHO", "BioPotentials", "Otolaryngology"]
modality = st.sidebar.selectbox("Select modality", MODALITIES)

uploaded = st.file_uploader(f"Upload {modality} file", type=None)

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    st.success(f"Saved upload to temporary file: {tmp_path}")

    result = {
        "summary": f"[Placeholder] No processor implemented for {modality} yet.",
        "metadata": {"file": uploaded.name},
        "images": [],
        "figures": [],
        "audio": None,
    }

    st.subheader("Processing summary")
    st.markdown(result.get("summary", "No summary produced."))

    if result.get("metadata"):
        st.subheader("Metadata")
        st.json(result["metadata"])

    if result.get("images"):
        st.subheader("Images")
        for img_path in result["images"]:
            st.image(img_path, use_column_width=True)

    if result.get("figures"):
        st.subheader("Plots")
        for fig in result["figures"]:
            st.pyplot(fig)

    if result.get("audio"):
        st.subheader("Audio preview")
        st.audio(result["audio"])

    st.markdown("---")
    if st.button("Get AI Insight"):
        prompt = f"""You are a clinical assistant.
Modality: {modality}
Filename: {uploaded.name}
Processing summary:
{result.get('summary','')}

Metadata:
{result.get('metadata',{})}

Provide concise clinical-relevant observations, uncertainties, and suggested next steps.
"""
        with st.spinner("Querying Gemini Flash..."):
            response = model.generate_content(prompt)
            insight = response.text

        st.subheader("AI Insight")
        st.write(insight)
