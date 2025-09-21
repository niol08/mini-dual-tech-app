import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import mne

# Try to import from label_map if available
try:
    from src.label_map import LABEL_REPHRASE
except ImportError:
    LABEL_REPHRASE = {}

def load_pickle_safely(file_path, fallback=None):
    """
    Safely load a pickle file with robust error handling for corrupted files.
    Returns fallback if loading fails.
    """
    if not os.path.exists(file_path):
        st.warning(f"Model file not found: {file_path}")
        return fallback
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.PickleError, EOFError, ValueError, KeyError) as e:
        st.error(f"Failed to load pickle file {file_path}: {str(e)}")
        st.info("This might be due to a corrupted or incompatible model file.")
        return fallback
    except Exception as e:
        st.error(f"Unexpected error loading {file_path}: {str(e)}")
        return fallback

def read_biosignal_file(uploaded_file):
    """
    Read various biosignal file formats and return processed data.
    Supports EDF, FIF, SET, BDF, CSV, TXT formats.
    """
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Handle different file formats
        if file_extension in ['edf', 'fif', 'set', 'bdf']:
            # Use MNE for EEG/biosignal formats
            if file_extension == 'edf':
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            elif file_extension == 'fif':
                raw = mne.io.read_raw_fif(tmp_path, preload=True, verbose=False)
            elif file_extension == 'set':
                raw = mne.io.read_raw_eeglab(tmp_path, preload=True, verbose=False)
            elif file_extension == 'bdf':
                raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose=False)
            
            # Extract data and convert to pandas DataFrame
            data = raw.get_data().T  # Transpose to get (samples, channels)
            column_names = raw.ch_names
            df = pd.DataFrame(data, columns=column_names)
            
        elif file_extension in ['csv', 'txt']:
            # Handle CSV/TXT files
            try:
                df = pd.read_csv(tmp_path)
            except:
                # Try different separators
                for sep in ['\t', ';', ' ']:
                    try:
                        df = pd.read_csv(tmp_path, sep=sep)
                        break
                    except:
                        continue
                else:
                    # If all separators fail, try reading as single column
                    with open(tmp_path, 'r') as f:
                        values = []
                        for line in f:
                            try:
                                values.append(float(line.strip()))
                            except ValueError:
                                continue
                    df = pd.DataFrame({'signal': values})
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def extract_biopotential_features(data_df):
    """
    Extract features from biopotential signals for AP/RP classification.
    """
    features = {}
    
    # Select numeric columns only
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.error("No numeric data found in the file.")
        return None
    
    # Use first numeric column if multiple available
    signal = data_df[numeric_cols[0]].values
    
    # Remove NaN values
    signal = signal[~np.isnan(signal)]
    
    if len(signal) == 0:
        st.error("No valid signal data found.")
        return None
    
    # Basic statistical features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['var'] = np.var(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['median'] = np.median(signal)
    features['skewness'] = pd.Series(signal).skew()
    features['kurtosis'] = pd.Series(signal).kurtosis()
    
    # Range and amplitude features
    features['range'] = features['max'] - features['min']
    features['rms'] = np.sqrt(np.mean(signal**2))
    
    # Peak detection features
    signal_diff = np.diff(signal)
    features['zero_crossings'] = len(np.where(np.diff(np.signbit(signal_diff)))[0])
    
    # Percentile features
    features['q25'] = np.percentile(signal, 25)
    features['q75'] = np.percentile(signal, 75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Energy and power features
    features['energy'] = np.sum(signal**2)
    features['power'] = features['energy'] / len(signal)
    
    # Rate of change features
    features['mean_abs_diff'] = np.mean(np.abs(signal_diff))
    features['max_abs_diff'] = np.max(np.abs(signal_diff))
    
    return pd.DataFrame([features])

def apply_label_mapping(prediction, confidence_scores=None):
    """
    Apply label mapping from label_map.py to remap results to AP/RP format.
    """
    # Default mapping if no custom mapping available
    default_mapping = {
        'action_potential': 'AP',
        'resting_potential': 'RP',
        'ap': 'AP',
        'rp': 'RP',
        '0': 'RP',  # Assuming class 0 is resting potential
        '1': 'AP',  # Assuming class 1 is action potential
        0: 'RP',
        1: 'AP'
    }
    
    # Use custom mapping if available, otherwise use default
    mapping = LABEL_REPHRASE if LABEL_REPHRASE else default_mapping
    
    # Convert prediction to string for consistent mapping
    pred_str = str(prediction).lower()
    
    # Apply mapping
    mapped_prediction = mapping.get(pred_str, mapping.get(prediction, prediction))
    
    # If we have confidence scores, also map them
    mapped_confidence = confidence_scores
    if confidence_scores and isinstance(confidence_scores, dict):
        mapped_confidence = {}
        for key, value in confidence_scores.items():
            mapped_key = mapping.get(str(key).lower(), mapping.get(key, key))
            mapped_confidence[mapped_key] = value
    
    return mapped_prediction, mapped_confidence

def process_biopotentials(uploaded_file):
    """
    Main function to process biopotential signals and classify AP/RP.
    Returns results with label mapping applied.
    """
    try:
        # Load models safely
        model_dir = Path("src/model")
        classifier_path = model_dir / "ap_rp_classifier.pkl"
        scaler_path = model_dir / "scaler.pkl"
        
        classifier = load_pickle_safely(classifier_path)
        scaler = load_pickle_safely(scaler_path)
        
        if classifier is None:
            return {
                "error": "AP/RP classifier model could not be loaded",
                "status": "failed",
                "details": "Please check if ap_rp_classifier.pkl exists and is not corrupted"
            }
        
        # Read the uploaded file
        data_df = read_biosignal_file(uploaded_file)
        if data_df is None:
            return {
                "error": "Could not read the uploaded file",
                "status": "failed"
            }
        
        # Extract features
        features_df = extract_biopotential_features(data_df)
        if features_df is None:
            return {
                "error": "Could not extract features from the signal",
                "status": "failed"
            }
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                features_scaled = scaler.transform(features_df)
                features_for_prediction = features_scaled
            except Exception as e:
                st.warning(f"Could not apply scaler: {e}. Using raw features.")
                features_for_prediction = features_df.values
        else:
            st.warning("Scaler not available. Using raw features.")
            features_for_prediction = features_df.values
        
        # Make prediction
        try:
            prediction = classifier.predict(features_for_prediction)[0]
            
            # Get prediction probabilities if available
            if hasattr(classifier, 'predict_proba'):
                probabilities = classifier.predict_proba(features_for_prediction)[0]
                # Map probabilities to class names
                classes = getattr(classifier, 'classes_', [0, 1])
                confidence_scores = dict(zip(classes, probabilities))
                confidence = max(probabilities)
            else:
                confidence_scores = None
                confidence = 0.8  # Default confidence if not available
            
            # Apply label mapping
            mapped_prediction, mapped_confidence = apply_label_mapping(prediction, confidence_scores)
            
            result = {
                "status": "success",
                "prediction": mapped_prediction,
                "original_prediction": prediction,
                "confidence": float(confidence),
                "confidence_scores": mapped_confidence,
                "features_extracted": len(features_df.columns),
                "signal_length": len(data_df),
                "file_name": uploaded_file.name,
                "model_info": {
                    "classifier_loaded": classifier is not None,
                    "scaler_loaded": scaler is not None,
                    "label_mapping_applied": True
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "status": "failed",
                "features_shape": features_for_prediction.shape if 'features_for_prediction' in locals() else None
            }
    
    except Exception as e:
        return {
            "error": f"Processing failed: {str(e)}",
            "status": "failed"
        }

# Streamlit interface
def main():
    st.title("BioPotentials AP/RP Classifier")
    st.write("Upload a biosignal file to classify Action Potential (AP) or Resting Potential (RP)")
    
    uploaded_file = st.file_uploader(
        "Choose a biosignal file",
        type=['edf', 'fif', 'set', 'bdf', 'csv', 'txt'],
        help="Supported formats: EDF, FIF, SET, BDF, CSV, TXT"
    )
    
    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size} bytes")
        
        if st.button("Classify Signal"):
            with st.spinner("Processing biosignal..."):
                result = process_biopotentials(uploaded_file)
            
            if result["status"] == "success":
                st.success("Classification completed!")
                
                # Display main results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", result["prediction"])
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.3f}")
                with col3:
                    st.metric("Features", result["features_extracted"])
                
                # Display confidence scores if available
                if result.get("confidence_scores"):
                    st.subheader("Class Probabilities")
                    for class_name, prob in result["confidence_scores"].items():
                        st.write(f"**{class_name}:** {prob:.4f}")
                
                # Show detailed results
                with st.expander("Detailed Results"):
                    st.json(result)
                    
            else:
                st.error("Processing failed!")
                st.error(result["error"])
                if "details" in result:
                    st.info(result["details"])

if __name__ == "__main__":
    main()
