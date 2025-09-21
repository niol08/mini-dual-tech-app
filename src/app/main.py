
import os
import streamlit as st
from dotenv import load_dotenv
from model_loader import HuggingFaceSpaceClient
from gemini import query_gemini_rest 

load_dotenv()

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")

if not HF_TOKEN:
    st.error("Hugging Face token required!")
    st.stop()

st.title("Biosignal Diagnostic Chatbot")

@st.cache_resource
def get_hf_client():
    return HuggingFaceSpaceClient(HF_TOKEN)

hf_client = get_hf_client()

FILE_TYPES = {
    "ECG": ["csv", "txt"],
    "EMG": ["csv", "txt"],
    "VAG": ["csv", "npy", "wav"],
    "PCG": ["wav"],
}

tabs = st.tabs(["ECG", "EMG", "VAG", "PCG"])

for tab, sig in zip(tabs, ["ECG", "EMG", "VAG", "PCG"]):
    with tab:
        st.header(f"{sig} Analysis")


        if sig == "ECG":
            with st.expander("ECG Data Requirements"):
                st.markdown(
                    "- Upload a `.csv` or `.txt` file containing **256 numeric values** (single row or single column).\n"
                    "- Example:\n"
                    "```csv\n0.12\n0.15\n-0.05\n...\n```"
                )
        elif sig == "VAG":
            with st.expander("VAG Data Requirements"):
                st.markdown(
                "- Upload a `.csv` file **with headers** containing the following 5 features:\n"
                "  - `rms_amplitude`\n"
                "  - `peak_frequency`\n"
                "  - `spectral_entropy`\n"
                "  - `zero_crossing_rate`\n"
                "  - `mean_frequency`\n"
                "- Example file content:\n"
                "```csv\n"
                "rms_amplitude,peak_frequency,spectral_entropy,zero_crossing_rate,mean_frequency\n"
                "1.02,20,-1890.34,0.001,39.7\n"
                "```"
                )
        elif sig == "EMG":
            with st.expander("EMG Data Requirements"):
                st.markdown(
                    "- Upload a `.txt` or `.csv` file containing **raw EMG signal samples**.\n"
                    "- The model expects **at least 1,000 values** (1-second window at 1 kHz sampling).\n"
                    "- You can provide:\n"
                    "  - A `.txt` file with one value per line.\n"
                    "  - A `.csv` file with a single column of numbers.\n\n"
                    "- Example `.txt` file:\n"
                    "```txt\n"
                    "0.034\n"
                    "0.056\n"
                    "-0.012\n"
                    "...\n"
                    "```"
                )
        elif sig == "PCG":
            with st.expander("PCG Data Requirements"):
                st.markdown(
                    "- Upload a `.wav` file containing a **single-channel (mono) PCG signal**.\n"
                    "- The model expects **at least 995 audio samples** (≈0.025s of heart sound at 44.1 kHz).\n"
                    "- Files longer than 995 samples will be **trimmed**; shorter ones will be **zero-padded**.\n"
                    "- Ensure the signal is **clean and preprocessed** (no ambient noise).\n\n"
                    "- Example `.wav` properties:\n"
                    "  - Mono (1 channel)\n"
                    "  - 44.1 kHz sampling rate\n"
                    "  - 16-bit PCM or float32\n"
                    "\n"
                    "_Note: Do not upload `.mp3`, `.flac`, or stereo files—they may fail to process properly._"
                )
        
        uploaded = st.file_uploader(
            f"Upload {sig} file",
            type=FILE_TYPES[sig],
            key=f"upload_{sig}"
        )

        if uploaded and st.button("Run Diagnostic", key=f"run_{sig}"):
            with st.spinner(f"Analyzing {sig} via HuggingFace Space..."):
                try:
                    if sig == "ECG":
                        label, human, conf = hf_client.predict_ecg(uploaded)
                    elif sig == "PCG":
                        label, human, conf = hf_client.predict_pcg(uploaded)
                    elif sig == "EMG":
                        human, conf = hf_client.predict_emg(uploaded)
                        label = human  
                    elif sig == "VAG":
                        label, human, conf = hf_client.predict_vag(uploaded)
                    
                    st.success(f"**{label} – {human}**\n\nConfidence: {conf:.2%}")
                    

                    if GEMINI_API_KEY:
                        try:
                            gnote = query_gemini_rest(sig, human, conf, GEMINI_API_KEY)
                            if gnote and not gnote.startswith("⚠️"):
                                st.markdown("### Gemini Insight")
                                st.write(gnote)
                            elif not gnote:
                                st.info("Gemini key missing – no explanation.")
                        except Exception as e:
                            st.warning(f"Gemini insight unavailable: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Make sure your HuggingFace Space is running and accessible.")

        else:
            if not uploaded:
                st.info("Upload a file to begin analysis.")

st.caption("© 2025 Biosignal Chatbot | Interface powered by streamlit cloud, ML by HuggingFace")