import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import tensorflow as tf
from PIL import Image
import cv2

def process_echo(file_path: str) -> Dict[str, Any]:
    """
    Process echocardiogram data using the neural network model weights.
    
    Args:
        file_path: Path to the echo data file
        
    Returns:
        Dictionary containing processing results
    """
    try:
        # Load the model weights
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
        weights_path = os.path.join(model_dir, 'model.weights.h5')
        config_path = os.path.join(model_dir, 'config.json')
        
        # Check if model files exist
        if not os.path.exists(weights_path):
            return {
                "summary": "Error: Model weights not found",
                "metadata": {"error": f"Weights file not found at {weights_path}"},
                "images": [],
                "figures": [],
                "audio": None
            }
        
        # Try to load configuration if available
        model_config = None
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        
        # Try to process the uploaded file
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Image file - likely echocardiogram image
            return process_echo_image(file_path, weights_path, model_config)
        elif file_ext in ['.csv', '.txt', '.xlsx']:
            # Data file - likely echo measurements
            return process_echo_data(file_path, weights_path, model_config)
        elif file_ext in ['.dcm', '.dicom']:
            # DICOM file - medical imaging format
            return process_echo_dicom(file_path, weights_path, model_config)
        else:
            # Try to handle as generic data
            return process_echo_generic(file_path, weights_path, model_config)
            
    except Exception as e:
        return {
            "summary": f"Error processing echo data: {str(e)}",
            "metadata": {"error": str(e), "file_path": file_path},
            "images": [],
            "figures": [],
            "audio": None
        }

def process_echo_image(file_path: str, weights_path: str, config: dict) -> Dict[str, Any]:
    """Process echocardiogram image"""
    try:
        # Load and display the image
        image = Image.open(file_path)
        img_array = np.array(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Echocardiogram')
        axes[0].axis('off')
        
        # Simple processing - convert to grayscale and enhance contrast
        if len(img_array.shape) == 3:
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = img_array
            
        # Apply histogram equalization for better contrast
        enhanced_image = cv2.equalizeHist(gray_image)
        
        axes[1].imshow(enhanced_image, cmap='gray')
        axes[1].set_title('Enhanced (Histogram Equalized)')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Basic image analysis
        mean_intensity = np.mean(gray_image)
        std_intensity = np.std(gray_image)
        
        summary = f"""Echocardiogram Image Analysis:
- Image dimensions: {img_array.shape}
- Mean intensity: {mean_intensity:.2f}
- Intensity std: {std_intensity:.2f}
- Image enhanced using histogram equalization
- Note: Deep learning model analysis requires specific model architecture"""
        
        metadata = {
            "file_type": "echo_image",
            "dimensions": list(img_array.shape),
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity),
            "model_weights_available": True
        }
        
        return {
            "summary": summary,
            "metadata": metadata,
            "images": [],
            "figures": [fig],
            "audio": None
        }
        
    except Exception as e:
        return {
            "summary": f"Error processing echo image: {str(e)}",
            "metadata": {"error": str(e)},
            "images": [],
            "figures": [],
            "audio": None
        }

def process_echo_data(file_path: str, weights_path: str, config: dict) -> Dict[str, Any]:
    """Process echocardiogram tabular data"""
    try:
        # Read data file
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path)
        else:
            data = pd.read_csv(file_path, delimiter='\t')
        
        # Basic analysis
        n_samples, n_features = data.shape
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Data overview
        axes[0, 0].text(0.1, 0.5, f"Dataset: {n_samples} samples, {n_features} features", 
                       transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Dataset Overview')
        axes[0, 0].axis('off')
        
        # Plot first few columns if numeric
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]
        if len(numeric_cols) > 0:
            for i, col in enumerate(numeric_cols):
                if i < 3:  # Plot up to 3 columns
                    axes[0, 1].plot(data[col][:min(100, len(data))], label=col, alpha=0.7)
            axes[0, 1].set_title('Echo Parameters (first 100 samples)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Correlation heatmap if we have numeric data
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Feature Correlations')
            axes[1, 0].set_xticks(range(len(numeric_cols)))
            axes[1, 0].set_yticks(range(len(numeric_cols)))
            axes[1, 0].set_xticklabels(numeric_cols, rotation=45)
            axes[1, 0].set_yticklabels(numeric_cols)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Distribution of first numeric column
        if len(numeric_cols) > 0:
            axes[1, 1].hist(data[numeric_cols[0]].dropna(), bins=30, alpha=0.7)
            axes[1, 1].set_title(f'Distribution of {numeric_cols[0]}')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        summary = f"""Echo Data Analysis:
- Dataset shape: {n_samples} samples × {n_features} features
- Numeric features: {len(numeric_cols)}
- Neural networitclo ne
git clone https://github.com/echonet/echonet-lvh.gitk weights available for advanced processing
- Basic statistical analysis performed"""
        
        metadata = {
            "file_type": "echo_data",
            "samples": int(n_samples),
            "features": int(n_features),
            "numeric_features": len(numeric_cols),
            "feature_names": list(data.columns),
            "model_weights_available": True
        }
        
        return {
            "summary": summary,
            "metadata": metadata,
            "images": [],
            "figures": [fig],
            "audio": None
        }
        
    except Exception as e:
        return {
            "summary": f"Error processing echo data: {str(e)}",
            "metadata": {"error": str(e)},
            "images": [],
            "figures": [],
            "audio": None
        }

def process_echo_dicom(file_path: str, weights_path: str, config: dict) -> Dict[str, Any]:
    """Process DICOM echocardiogram files"""
    try:
        # Try to read DICOM file
        try:
            import pydicom
            ds = pydicom.dcmread(file_path)
            
            # Extract image data
            img_data = ds.pixel_array
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original DICOM image
            axes[0].imshow(img_data, cmap='gray')
            axes[0].set_title('DICOM Echocardiogram')
            axes[0].axis('off')
            
            # Enhanced version
            enhanced = cv2.equalizeHist(img_data.astype(np.uint8))
            axes[1].imshow(enhanced, cmap='gray')
            axes[1].set_title('Enhanced Image')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            summary = f"""DICOM Echo Analysis:
- Image dimensions: {img_data.shape}
- Pixel intensity range: {img_data.min()} - {img_data.max()}
- Patient ID: {getattr(ds, 'PatientID', 'N/A')}
- Study Date: {getattr(ds, 'StudyDate', 'N/A')}
- Modality: {getattr(ds, 'Modality', 'N/A')}"""
            
            metadata = {
                "file_type": "echo_dicom",
                "dimensions": list(img_data.shape),
                "pixel_range": [int(img_data.min()), int(img_data.max())],
                "patient_id": str(getattr(ds, 'PatientID', 'N/A')),
                "study_date": str(getattr(ds, 'StudyDate', 'N/A')),
                "modality": str(getattr(ds, 'Modality', 'N/A'))
            }
            
        except ImportError:
            summary = "DICOM file detected but pydicom library not available for processing"
            metadata = {"error": "pydicom not installed"}
            fig = plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'DICOM processing requires pydicom library', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('DICOM Processing Unavailable')
            
        return {
            "summary": summary,
            "metadata": metadata,
            "images": [],
            "figures": [fig],
            "audio": None
        }
        
    except Exception as e:
        return {
            "summary": f"Error processing DICOM echo: {str(e)}",
            "metadata": {"error": str(e)},
            "images": [],
            "figures": [],
            "audio": None
        }

def process_echo_generic(file_path: str, weights_path: str, config: dict) -> Dict[str, Any]:
    """Generic echo file processing"""
    try:
        # Try to read as text/data file
        with open(file_path, 'r') as f:
            content = f.read(1000)  # First 1000 chars
        
        file_size = os.path.getsize(file_path)
        
        summary = f"""Generic Echo File Analysis:
- File size: {file_size} bytes
- Content preview: {content[:200]}...
- Neural network weights available for custom processing
- Consider converting to supported format (CSV, image, DICOM)"""
        
        metadata = {
            "file_type": "echo_generic",
            "file_size": file_size,
            "content_preview": content[:200],
            "model_weights_available": True
        }
        
        return {
            "summary": summary,
            "metadata": metadata,
            "images": [],
            "figures": [],
            "audio": None
        }
        
    except Exception as e:
        return {
            "summary": f"Error processing generic echo file: {str(e)}",
            "metadata": {"error": str(e)},
            "images": [],
            "figures": [],
            "audio": None
        }
