import os
import gdown
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

def debug_environment():
    import os
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    if os.path.exists('models'):
        print(f"Models directory exists: {os.listdir('models')}")
    else:
        print("Models directory does not exist")


def extract_file_id_from_url(url):
    """Extract file ID from Google Drive URL"""
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1].split("&")[0]
    return url  

def get_model_urls():
    try:
     return{
      "../models/MLII-latest.keras": st.secrets["ECG_MODEL_URL"],
      "../models/pcg_model.h5": st.secrets["PCG_MODEL_URL"],
      "../models/emg_model.h5": st.secrets["EMG_MODEL_URL"],
      "../models/vag_feature_classifier.pkl": st.secrets["VAG_MODEL_URL"]
     }
    except:
     return{
      "../models/MLII-latest.keras": os.getenv("ECG_MODEL_URL", ""),
      "../models/pcg_model.h5": os.getenv("PCG_MODEL_URL", ""),
      "../models/emg_model.h5": os.getenv("EMG_MODEL_URL", ""),
      "../models/vag_feature_classifier.pkl": os.getenv("VAG_MODEL_URL", "") 
     }

def download_from_gdrive(url, output_path):
    """Download file from Google Drive using gdown"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
   
    file_id = extract_file_id_from_url(url)
    
    
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(download_url, output_path, quiet=False)

def ensure_models_downloaded():
    """Download models if they don't exist locally"""
    
    debug_environment()
    
    model_urls = get_model_urls()
    
    for local_path, url in model_urls.items():
        if not url:
            print(f"No URL found for {local_path}")
            continue
            
        if not os.path.exists(local_path):
            print(f"Downloading {local_path}...")
            try:
                download_from_gdrive(url, local_path)
                print(f"Downloaded {local_path}")
            except Exception as e:
                print(f"Failed to download {local_path}: {e}")
        else:
            print(f"{local_path} already exists")

if __name__ == "__main__":
    ensure_models_downloaded()