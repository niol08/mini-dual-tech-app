import os
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.processors.ct import process_ct
from src.processors.erp import process_erp
from src.processors.ecg import process_ecg
from src.processors.emg import process_emg

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file. or your cloud environment")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Biosignal & Imaging", layout="wide")
st.title("Biosignal & Imaging")

MODALITIES = ["BioPotentials", "ECG", "EMG", "CT", "Speech", "Otolaryngology"]
modality = st.sidebar.selectbox("Select modality", MODALITIES)

uploaded = st.file_uploader(f"Upload {modality} file", type=None)

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    st.success(f"Saved upload to temporary file: {tmp_path}")

    # Process based on selected modality
    try:
        if modality == "BioPotentials":
            result = process_erp(tmp_path)  # ERP uses the direct ERP processor
        elif modality == "ECG":
            result = process_ecg(tmp_path)  # ECG uses HuggingFace model loader
        elif modality == "EMG":
            result = process_emg(tmp_path)  # EMG uses HuggingFace model loader
        elif modality == "CT":
            result = process_ct(tmp_path)
        else:
            # Placeholder for other modalities
            result = {
                "summary": f"[Placeholder] No processor implemented for {modality} yet.",
                "metadata": {"file": uploaded.name},
                "images": [],
                "figures": [],
                "audio": None,
            }
    except Exception as e:
        result = {
            "summary": f"Error processing {modality} file: {str(e)}",
            "metadata": {"error": str(e), "file": uploaded.name},
            "images": [],
            "figures": [],
            "audio": None,
        }
        st.error(f"Processing error: {str(e)}")

    # Clean up temporary file
    try:
        os.unlink(tmp_path)
    except:
        pass

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
