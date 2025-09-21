import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import sys

# Import label mapping
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from label_map import apply_label_mapping
except ImportError:
    def apply_label_mapping(classification, confidence, processor_type='biopotentials'):
        return {'mapped_label': classification, 'confidence': confidence, 'mapping_applied': False}

def create_mock_biopotential_classifier():
    """
    Create a mock AP/RP classifier for demonstration when actual model files are missing.
    This simulates Action Potential (AP) vs Resting Potential (RP) classification.
    """
    class MockAPRPClassifier:
        def __init__(self):
            self.classes_ = ['0', '1']  # 0 = RP, 1 = AP
            
        def predict(self, X):
            """Mock prediction based on signal characteristics"""
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            predictions = []
            for sample in X:
                # Simple mock logic: if mean voltage > -50mV, predict AP (1), else RP (0)
                # This simulates the basic biophysics where action potentials are more positive
                mean_voltage = np.mean(sample)
                pred = '1' if mean_voltage > -50.0 else '0'
                predictions.append(pred)
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            """Mock probability prediction for AP vs RP"""
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            probabilities = []
            for sample in X:
                mean_voltage = np.mean(sample)
                # Create probabilities based on voltage level
                # More positive = higher AP probability
                if mean_voltage > -40:  # Strong AP indicator
                    prob_ap = 0.85
                elif mean_voltage > -50:  # Moderate AP indicator  
                    prob_ap = 0.65
                elif mean_voltage > -60:  # Weak AP indicator
                    prob_ap = 0.45
                else:  # RP indicator
                    prob_ap = 0.25
                    
                prob_rp = 1.0 - prob_ap
                probabilities.append([prob_rp, prob_ap])  # [P(RP), P(AP)]
            
            return np.array(probabilities)
    
    return MockAPRPClassifier()

def load_pickle_safely(file_path, fallback=None):
    """
    Safely load a pickle file with robust error handling.
    Returns fallback if loading fails.
    """
    if not os.path.exists(file_path):
        return fallback
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load {file_path}: {e}")
        return fallback

def read_biosignal_file(uploaded_file):
    """
    Read various biosignal file formats and return processed data.
    Supports CSV, TXT formats for voltage/potential measurements.
    """
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Handle different file formats
        if file_extension == 'csv':
            data = pd.read_csv(tmp_path)
        elif file_extension == 'txt':
            # Try different delimiters
            try:
                data = pd.read_csv(tmp_path, delimiter='\t')
            except:
                try:
                    data = pd.read_csv(tmp_path, delimiter=' ')
                except:
                    # Assume single column of voltage values
                    with open(tmp_path, 'r') as f:
                        values = [float(line.strip()) for line in f if line.strip()]
                    data = pd.DataFrame({'voltage': values})
        else:
            # Try to read as generic CSV
            data = pd.read_csv(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return data
        
    except Exception as e:
        st.error(f"Error reading biosignal file: {e}")
        return None

def extract_biopotential_features(data_df):
    """
    Extract features relevant to AP/RP classification from biosignal data.
    Features focus on voltage characteristics that distinguish action vs resting potentials.
    """
    try:
        # Find voltage column (common names)
        voltage_col = None
        for col in data_df.columns:
            if any(term in col.lower() for term in ['voltage', 'potential', 'mv', 'v']):
                voltage_col = col
                break
        
        # If no specific voltage column found, use the last numeric column
        if voltage_col is None:
            numeric_cols = data_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                voltage_col = numeric_cols[-1]  # Assume last column is voltage
            else:
                raise ValueError("No numeric voltage data found")
        
        voltage_data = data_df[voltage_col].dropna()
        
        if len(voltage_data) == 0:
            raise ValueError("No valid voltage data found")
        
        # Extract AP/RP relevant features
        features = {}
        
        # Basic statistical features
        features['mean_voltage'] = np.mean(voltage_data)
        features['std_voltage'] = np.std(voltage_data)
        features['min_voltage'] = np.min(voltage_data)
        features['max_voltage'] = np.max(voltage_data)
        features['range_voltage'] = features['max_voltage'] - features['min_voltage']
        
        # AP-specific features
        features['peak_voltage'] = np.max(voltage_data)  # Action potentials have high peaks
        features['depolarization_extent'] = max(0, features['peak_voltage'] + 70)  # From resting ~-70mV
        
        # RP-specific features  
        features['baseline_voltage'] = np.percentile(voltage_data, 25)  # Lower quartile as baseline
        features['time_below_minus50'] = np.sum(voltage_data < -50) / len(voltage_data)  # Time in RP range
        
        # Dynamic features
        voltage_diff = np.diff(voltage_data)
        features['max_depolarization_rate'] = np.max(voltage_diff) if len(voltage_diff) > 0 else 0
        features['max_repolarization_rate'] = np.abs(np.min(voltage_diff)) if len(voltage_diff) > 0 else 0
        
        # Threshold crossing features
        features['crossings_minus40'] = np.sum(np.diff(voltage_data > -40)) // 2  # AP threshold crossings
        features['crossings_minus60'] = np.sum(np.diff(voltage_data > -60)) // 2  # RP threshold crossings
        
        return pd.DataFrame([features])
        
    except Exception as e:
        st.error(f"Error extracting biopotential features: {e}")
        return None

def process_biopotentials(uploaded_file):
    """
    Main function to process biopotential signals and classify AP/RP.
    Returns results in format expected by ERP wrapper with AP/RP classification.
    """
    try:
        # Load models - use relative path from processors directory
        model_dir = Path(os.path.dirname(__file__)) / ".." / "model"
        classifier_path = model_dir / "ap_rp_classifier.pkl"
        scaler_path = model_dir / "scaler.pkl"
        
        # Try to load real models, fallback to mock
        classifier = load_pickle_safely(classifier_path)
        scaler = load_pickle_safely(scaler_path)
        
        if classifier is None:
            st.info("AP/RP classifier model not found. Using demonstration classifier.")
            classifier = create_mock_biopotential_classifier()
            scaler = None
        
        # Read the uploaded file
        data_df = read_biosignal_file(uploaded_file)
        if data_df is None:
            return {
                "error": "Could not read the uploaded biopotential file",
                "status": "failed",
                "metadata": {
                    "models_loaded": False,
                    "dominant_class": None,
                    "class_distribution": {},
                    "processed_samples": 0,
                    "error": "File reading failed"
                },
                "summary": "BioPotentials: Could not read uploaded file"
            }
        
        # Extract features
        features_df = extract_biopotential_features(data_df)
        if features_df is None:
            return {
                "error": "Could not extract biopotential features from signal",
                "status": "failed", 
                "metadata": {
                    "models_loaded": False,
                    "dominant_class": None,
                    "class_distribution": {},
                    "processed_samples": 0,
                    "error": "Feature extraction failed"
                },
                "summary": "BioPotentials: Feature extraction failed"
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
            features_for_prediction = features_df.values
        
        # Make prediction
        try:
            prediction = classifier.predict(features_for_prediction)[0]
            
            # Get prediction probabilities if available
            if hasattr(classifier, 'predict_proba'):
                probabilities = classifier.predict_proba(features_for_prediction)[0]
                # Map probabilities to class names
                classes = getattr(classifier, 'classes_', ['0', '1'])
                confidence_scores = dict(zip(classes, probabilities))
                confidence = max(probabilities)
            else:
                confidence_scores = {str(prediction): 0.8}
                confidence = 0.8
            
            # Apply label mapping for AP/RP
            mapped_result = apply_label_mapping(prediction, confidence, processor_type='biopotentials')
            
            # Create result in format expected by ERP wrapper
            result = {
                "status": "success",
                "prediction": mapped_result['mapped_label'],
                "original_prediction": prediction,
                "confidence": float(mapped_result['confidence']),
                "confidence_scores": confidence_scores,
                "features_extracted": len(features_df.columns),
                "signal_length": len(data_df),
                "file_name": uploaded_file.name,
                "models_loaded": True,
                "dominant_class": prediction,
                "class_distribution": confidence_scores,
                # Metadata structure expected by ERP wrapper
                "metadata": {
                    "models_loaded": True,
                    "dominant_class": prediction,
                    "class_distribution": confidence_scores,
                    "processed_samples": 1,
                    "classification": mapped_result['mapped_label'],
                    "confidence": float(mapped_result['confidence']),
                    "original_classification": prediction,
                    "processor_type": "biopotentials",
                    "ap_rp_result": {
                        "class": "AP" if prediction == '1' else "RP", 
                        "full_name": "Action Potential" if prediction == '1' else "Resting Potential",
                        "confidence": float(confidence)
                    }
                },
                # Summary for display
                "summary": f"""BioPotentials AP/RP Analysis Complete:
- Classification: {mapped_result['mapped_label']} (confidence: {mapped_result['confidence']:.3f})
- Result: {'Action Potential (AP)' if prediction == '1' else 'Resting Potential (RP)'}
- Features extracted: {len(features_df.columns)}
- Signal length: {len(data_df)} samples
- Model status: {'Real model' if scaler else 'Demo classifier'}
- Confidence scores: {confidence_scores}"""
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"AP/RP prediction failed: {str(e)}",
                "status": "failed",
                "metadata": {
                    "models_loaded": False,
                    "dominant_class": None,
                    "class_distribution": {},
                    "processed_samples": 0,
                    "error": str(e)
                },
                "summary": f"BioPotentials prediction failed: {str(e)}"
            }
    
    except Exception as e:
        return {
            "error": f"BioPotentials processing failed: {str(e)}",
            "status": "failed",
            "metadata": {
                "models_loaded": False,
                "dominant_class": None,
                "class_distribution": {},
                "processed_samples": 0,
                "error": str(e)
            },
            "summary": f"BioPotentials processing failed: {str(e)}"
        }

def main():
    """Streamlit interface for standalone testing"""
    st.title("BioPotentials AP/RP Classifier")
    st.write("Upload a biosignal file to classify Action Potential (AP) vs Resting Potential (RP)")
    
    uploaded_file = st.file_uploader(
        "Choose a biosignal file",
        type=['csv', 'txt'],
        help="Supported formats: CSV, TXT with voltage/potential measurements"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing biopotential signal..."):
            result = process_biopotentials(uploaded_file)
            
            if result["status"] == "success":
                st.success("✅ Classification successful!")
                st.write(f"**Prediction:** {result['prediction']}")
                st.write(f"**Confidence:** {result['confidence']:.3f}")
                st.write(f"**AP/RP Result:** {result['metadata']['ap_rp_result']['full_name']}")
                
                # Show confidence scores
                if result['confidence_scores']:
                    st.write("**Class Probabilities:**")
                    for class_name, prob in result['confidence_scores'].items():
                        class_label = "Action Potential (AP)" if class_name == '1' else "Resting Potential (RP)"
                        st.write(f"- {class_label}: {prob:.3f}")
            else:
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
