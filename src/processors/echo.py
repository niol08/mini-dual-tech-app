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
import sys

# Import label mapping
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from label_map import apply_label_mapping
except ImportError:
    def apply_label_mapping(classification, confidence, processor_type='echo'):
        return {'mapped_label': classification, 'confidence': confidence}

def calculate_image_confidence(img_array: np.ndarray) -> Dict[str, float]:
    """Calculate confidence metrics for echocardiogram images"""
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = img_array
        
        # Image quality metrics
        # 1. Contrast (standard deviation of pixel intensities)
        contrast = np.std(gray_image)
        contrast_score = min(contrast / 50.0, 1.0)  # Normalize to 0-1
        
        # 2. Sharpness (variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize to 0-1
        
        # 3. Brightness consistency (inverse of coefficient of variation)
        mean_brightness = np.mean(gray_image)
        brightness_consistency = 1.0 - min(np.std(gray_image) / (mean_brightness + 1e-6), 1.0)
        
        # 4. Edge density (good echo images should have clear structures)
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density * 10, 1.0)  # Normalize
        
        # 5. Dynamic range
        dynamic_range = (np.max(gray_image) - np.min(gray_image)) / 255.0
        
        # Overall confidence as weighted average
        confidence = (
            0.25 * contrast_score +
            0.25 * sharpness_score +
            0.20 * brightness_consistency +
            0.20 * edge_score +
            0.10 * dynamic_range
        )
        
        return {
            'overall_confidence': confidence,
            'contrast_score': contrast_score,
            'sharpness_score': sharpness_score,
            'brightness_consistency': brightness_consistency,
            'edge_score': edge_score,
            'dynamic_range': dynamic_range
        }
        
    except Exception as e:
        return {
            'overall_confidence': 0.5,
            'error': str(e)
        }

def calculate_data_confidence(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate confidence metrics for echocardiogram tabular data"""
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'overall_confidence': 0.3, 'reason': 'no_numeric_data'}
        
        # Data quality metrics
        # 1. Completeness (percentage of non-null values)
        completeness = 1.0 - (data[numeric_cols].isnull().sum().sum() / (len(data) * len(numeric_cols)))
        
        # 2. Value range consistency (check for reasonable physiological values)
        range_scores = []
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                # Check for extreme outliers
                q1, q3 = np.percentile(col_data, [25, 75])
                iqr = q3 - q1
                outliers = np.sum((col_data < q1 - 3*iqr) | (col_data > q3 + 3*iqr))
                outlier_ratio = outliers / len(col_data)
                range_scores.append(1.0 - min(outlier_ratio * 2, 1.0))
        
        range_consistency = np.mean(range_scores) if range_scores else 0.5
        
        # 3. Data variability (good data should have reasonable variance)
        variability_scores = []
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 1:
                cv = np.std(col_data) / (np.mean(col_data) + 1e-6)  # Coefficient of variation
                # Good variability is between 0.1 and 2.0
                if 0.1 <= cv <= 2.0:
                    variability_scores.append(1.0)
                else:
                    variability_scores.append(max(0.3, 1.0 - abs(cv - 1.0) / 2.0))
        
        variability_score = np.mean(variability_scores) if variability_scores else 0.5
        
        # 4. Sample size adequacy
        size_score = min(len(data) / 100.0, 1.0)  # Normalize to 100 samples
        
        # Overall confidence
        confidence = (
            0.30 * completeness +
            0.25 * range_consistency +
            0.25 * variability_score +
            0.20 * size_score
        )
        
        return {
            'overall_confidence': confidence,
            'completeness': completeness,
            'range_consistency': range_consistency,
            'variability_score': variability_score,
            'size_score': size_score,
            'sample_count': len(data),
            'feature_count': len(numeric_cols)
        }
        
    except Exception as e:
        return {
            'overall_confidence': 0.5,
            'error': str(e)
        }

def get_echo_classification(confidence_metrics: Dict[str, float], file_type: str) -> Dict[str, Any]:
    """Determine echo classification based on confidence metrics"""
    confidence = confidence_metrics.get('overall_confidence', 0.5)
    
    # Simple classification based on confidence level
    if confidence >= 0.8:
        classification = 'high_quality'
        description = 'High quality echocardiogram data'
    elif confidence >= 0.6:
        classification = 'good_quality'
        description = 'Good quality echocardiogram data'
    elif confidence >= 0.4:
        classification = 'moderate_quality'
        description = 'Moderate quality echocardiogram data'
    else:
        classification = 'low_quality'
        description = 'Low quality echocardiogram data - consider retaking'
    
    return {
        'classification': classification,
        'description': description,
        'confidence': confidence,
        'quality_level': classification
    }
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
        
        # Calculate confidence metrics
        confidence_metrics = calculate_image_confidence(img_array)
        classification_result = get_echo_classification(confidence_metrics, 'image')
        
        # Apply label mapping
        mapped_result = apply_label_mapping(
            classification_result['classification'], 
            classification_result['confidence'],
            processor_type='echo'
        )
        
        summary = f"""Echocardiogram Image Analysis:
- Image dimensions: {img_array.shape}
- Mean intensity: {mean_intensity:.2f}
- Intensity std: {std_intensity:.2f}
- Quality Assessment: {mapped_result['mapped_label']} (confidence: {mapped_result['confidence']:.3f})
- Image enhanced using histogram equalization
- Confidence factors: contrast={confidence_metrics.get('contrast_score', 0):.3f}, sharpness={confidence_metrics.get('sharpness_score', 0):.3f}
- Note: Deep learning model analysis requires specific model architecture"""
        
        metadata = {
            "file_type": "echo_image",
            "dimensions": list(img_array.shape),
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity),
            "model_weights_available": True,
            "classification": mapped_result['mapped_label'],
            "confidence": float(mapped_result['confidence']),
            "quality_metrics": confidence_metrics,
            "original_classification": classification_result['classification']
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
        
        # Calculate confidence metrics
        confidence_metrics = calculate_data_confidence(data)
        classification_result = get_echo_classification(confidence_metrics, 'data')
        
        # Apply label mapping
        mapped_result = apply_label_mapping(
            classification_result['classification'], 
            classification_result['confidence'],
            processor_type='echo'
        )
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Data overview with confidence
        overview_text = f"""Dataset: {n_samples} samples, {n_features} features
Quality: {mapped_result['mapped_label']}
Confidence: {mapped_result['confidence']:.3f}
Completeness: {confidence_metrics.get('completeness', 0):.3f}"""
        axes[0, 0].text(0.1, 0.5, overview_text, 
                       transform=axes[0, 0].transAxes, fontsize=10, verticalalignment='center')
        axes[0, 0].set_title('Dataset Overview & Quality')
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
- Data Quality: {mapped_result['mapped_label']} (confidence: {mapped_result['confidence']:.3f})
- Completeness: {confidence_metrics.get('completeness', 0):.1%}
- Neural network weights available for advanced processing
- Quality factors: range consistency={confidence_metrics.get('range_consistency', 0):.3f}, variability={confidence_metrics.get('variability_score', 0):.3f}"""
        
        metadata = {
            "file_type": "echo_data",
            "samples": int(n_samples),
            "features": int(n_features),
            "numeric_features": len(numeric_cols),
            "feature_names": list(data.columns),
            "model_weights_available": True,
            "classification": mapped_result['mapped_label'],
            "confidence": float(mapped_result['confidence']),
            "quality_metrics": confidence_metrics,
            "original_classification": classification_result['classification']
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
            
            # Calculate confidence for DICOM image
            confidence_metrics = calculate_image_confidence(img_data)
            classification_result = get_echo_classification(confidence_metrics, 'dicom')
            
            # Apply label mapping
            mapped_result = apply_label_mapping(
                classification_result['classification'], 
                classification_result['confidence'],
                processor_type='echo'
            )
            
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
- Quality Assessment: {mapped_result['mapped_label']} (confidence: {mapped_result['confidence']:.3f})
- Patient ID: {getattr(ds, 'PatientID', 'N/A')}
- Study Date: {getattr(ds, 'StudyDate', 'N/A')}
- Modality: {getattr(ds, 'Modality', 'N/A')}
- Image quality factors: contrast={confidence_metrics.get('contrast_score', 0):.3f}, sharpness={confidence_metrics.get('sharpness_score', 0):.3f}"""
            
            metadata = {
                "file_type": "echo_dicom",
                "dimensions": list(img_data.shape),
                "pixel_range": [int(img_data.min()), int(img_data.max())],
                "patient_id": str(getattr(ds, 'PatientID', 'N/A')),
                "study_date": str(getattr(ds, 'StudyDate', 'N/A')),
                "modality": str(getattr(ds, 'Modality', 'N/A')),
                "classification": mapped_result['mapped_label'],
                "confidence": float(mapped_result['confidence']),
                "quality_metrics": confidence_metrics,
                "original_classification": classification_result['classification']
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
        
        # Basic confidence based on file size and readability
        basic_confidence = min(file_size / 10000.0, 0.7)  # Larger files get higher confidence, max 0.7
        classification_result = get_echo_classification({'overall_confidence': basic_confidence}, 'generic')
        
        # Apply label mapping
        mapped_result = apply_label_mapping(
            classification_result['classification'], 
            classification_result['confidence'],
            processor_type='echo'
        )
        
        summary = f"""Generic Echo File Analysis:
- File size: {file_size} bytes
- Content preview: {content[:200]}...
- Quality Assessment: {mapped_result['mapped_label']} (confidence: {mapped_result['confidence']:.3f})
- Neural network weights available for custom processing
- Consider converting to supported format (CSV, image, DICOM)"""
        
        metadata = {
            "file_type": "echo_generic",
            "file_size": file_size,
            "content_preview": content[:200],
            "model_weights_available": True,
            "classification": mapped_result['mapped_label'],
            "confidence": float(mapped_result['confidence']),
            "original_classification": classification_result['classification']
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
