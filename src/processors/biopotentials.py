import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

def process_biopotentials(file_path: str) -> Dict[str, Any]:
    """
    Process biopotential signals using AP/RP classifier and scaler.
    
    Args:
        file_path: Path to the biopotential data file
        
    Returns:
        Dictionary containing processing results
    """
    try:
        # Load the classifier and scaler
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
        classifier_path = os.path.join(model_dir, 'ap_rp_classifier.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        # Check if model files exist
        if not os.path.exists(classifier_path):
            return {
                "summary": "Error: AP/RP classifier model not found",
                "metadata": {"error": f"Model file not found at {classifier_path}"},
                "images": [],
                "figures": [],
                "audio": None
            }
            
        if not os.path.exists(scaler_path):
            return {
                "summary": "Error: Scaler model not found", 
                "metadata": {"error": f"Scaler file not found at {scaler_path}"},
                "images": [],
                "figures": [],
                "audio": None
            }
        
        # Load models
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Try to read the data file
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path)
        elif file_path.lower().endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t')
        else:
            # Try to read as generic text file
            try:
                data = pd.read_csv(file_path)
            except:
                data = np.loadtxt(file_path)
                data = pd.DataFrame(data)
        
        # Basic data info
        n_samples = len(data)
        n_features = data.shape[1] if len(data.shape) > 1 else 1
        
        # Create a simple visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot raw signal (first column or first few columns)
        if isinstance(data, pd.DataFrame):
            signal_data = data.iloc[:, 0] if data.shape[1] > 0 else data.iloc[:, :]
        else:
            signal_data = data[:, 0] if len(data.shape) > 1 else data
            
        axes[0].plot(signal_data[:min(1000, len(signal_data))])  # Plot first 1000 points
        axes[0].set_title('Raw Biopotential Signal')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        # Simple feature extraction for classification
        if isinstance(data, pd.DataFrame) and data.shape[1] >= scaler.n_features_in_:
            # Use first n features that match scaler expectation
            features = data.iloc[:100, :scaler.n_features_in_].values  # First 100 samples
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict using classifier
            predictions = classifier.predict(features_scaled)
            probabilities = classifier.predict_proba(features_scaled) if hasattr(classifier, 'predict_proba') else None
            
            # Plot predictions
            axes[1].plot(predictions[:min(100, len(predictions))])
            axes[1].set_title('AP/RP Classification Results')
            axes[1].set_xlabel('Sample')
            axes[1].set_ylabel('Predicted Class')
            axes[1].grid(True)
            
            # Calculate statistics
            unique_classes, counts = np.unique(predictions, return_counts=True)
            class_distribution = dict(zip(unique_classes, counts))
            
            summary = f"""Biopotential Analysis Complete:
- Samples processed: {n_samples}
- Features: {n_features}
- Classification performed on {len(features)} samples
- Class distribution: {class_distribution}
- Dominant class: {unique_classes[np.argmax(counts)]}"""

            metadata = {
                "file_type": "biopotential",
                "samples": int(n_samples),
                "features": int(n_features),
                "processed_samples": int(len(features)),
                "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
                "dominant_class": str(unique_classes[np.argmax(counts)])
            }
            
        else:
            axes[1].text(0.5, 0.5, 'Insufficient features for classification', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Classification Not Available')
            
            summary = f"""Biopotential Data Loaded:
- Samples: {n_samples}
- Features: {n_features}
- Note: Insufficient features for AP/RP classification (need {scaler.n_features_in_} features)"""

            metadata = {
                "file_type": "biopotential",
                "samples": int(n_samples),
                "features": int(n_features),
                "status": "insufficient_features_for_classification"
            }
        
        plt.tight_layout()
        
        return {
            "summary": summary,
            "metadata": metadata,
            "images": [],
            "figures": [fig],
            "audio": None
        }
        
    except Exception as e:
        return {
            "summary": f"Error processing biopotential data: {str(e)}",
            "metadata": {"error": str(e), "file_path": file_path},
            "images": [],
            "figures": [],
            "audio": None
        }
