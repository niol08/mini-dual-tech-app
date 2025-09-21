import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from PIL import Image
import sys

def process_ct(file_path: str) -> Dict[str, Any]:
    """
    Process CT scan data using the Swin transformer model.
    
    Args:
        file_path: Path to the CT scan file
        
    Returns:
        Dictionary containing processing results
    """
    try:
        # Import the Swin wrapper
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'CT')
        sys.path.append(model_dir)
        
        try:
            from swin_wrapper import SwinHFClassifier
            
            # Initialize the classifier
            classifier = SwinHFClassifier()
            
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.dcm', '.dicom']:
                return process_ct_dicom(file_path, classifier)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return process_ct_image(file_path, classifier)
            else:
                return process_ct_unknown(file_path)
                
        except ImportError as e:
            return {
                "summary": f"CT processing requires additional dependencies: {str(e)}",
                "metadata": {"error": f"Import error: {str(e)}", "missing_dependencies": ["torch", "transformers", "pydicom"]},
                "images": [],
                "figures": [],
                "audio": None
            }
        except Exception as e:
            return {
                "summary": f"Error initializing CT processor: {str(e)}",
                "metadata": {"error": str(e)},
                "images": [],
                "figures": [],
                "audio": None
            }
            
    except Exception as e:
        return {
            "summary": f"Error processing CT data: {str(e)}",
            "metadata": {"error": str(e), "file_path": file_path},
            "images": [],
            "figures": [],
            "audio": None
        }

def process_ct_dicom(file_path: str, classifier) -> Dict[str, Any]:
    """Process DICOM CT scan"""
    try:
        # Use the classifier to predict
        result = classifier.predict_single(file_path)
        
        # Also get the image for visualization
        image = classifier._load_image(file_path)
        img_array = np.array(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original CT scan
        axes[0].imshow(img_array)
        axes[0].set_title('CT Scan (DICOM)')
        axes[0].axis('off')
        
        # Classification result visualization
        axes[1].text(0.5, 0.7, f"Prediction: {result['label_name']}", 
                    ha='center', va='center', transform=axes[1].transAxes, 
                    fontsize=14, fontweight='bold')
        axes[1].text(0.5, 0.5, f"Confidence: {result['confidence']:.3f}", 
                    ha='center', va='center', transform=axes[1].transAxes, 
                    fontsize=12)
        axes[1].text(0.5, 0.3, f"Model: {result['ai_insight']}", 
                    ha='center', va='center', transform=axes[1].transAxes, 
                    fontsize=10)
        axes[1].set_title('Classification Result')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Load DICOM metadata if possible
        try:
            import pydicom
            ds = pydicom.dcmread(file_path)
            dicom_info = {
                "patient_id": str(getattr(ds, 'PatientID', 'N/A')),
                "study_date": str(getattr(ds, 'StudyDate', 'N/A')),
                "modality": str(getattr(ds, 'Modality', 'N/A')),
                "slice_thickness": str(getattr(ds, 'SliceThickness', 'N/A'))
            }
        except:
            dicom_info = {"status": "DICOM metadata not available"}
        
        summary = f"""CT DICOM Analysis Complete:
- Classification: {result['label_name']}
- Confidence: {result['confidence']:.3f}
- Image dimensions: {img_array.shape}
- Model: Swin Transformer (Hugging Face)
- Analysis: {result['ai_insight']}"""
        
        metadata = {
            "file_type": "ct_dicom",
            "classification": result['label_name'],
            "confidence": float(result['confidence']),
            "dimensions": list(img_array.shape),
            "model": "swin_transformer",
            "dicom_info": dicom_info
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
            "summary": f"Error processing CT DICOM: {str(e)}",
            "metadata": {"error": str(e)},
            "images": [],
            "figures": [],
            "audio": None
        }

def process_ct_image(file_path: str, classifier) -> Dict[str, Any]:
    """Process regular image file as CT scan"""
    try:
        # Use the classifier to predict
        result = classifier.predict_single(file_path)
        
        # Load image for visualization
        image = Image.open(file_path).convert('RGB')
        img_array = np.array(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('CT Image')
        axes[0].axis('off')
        
        # Classification result
        axes[1].text(0.5, 0.7, f"Prediction: {result['label_name']}", 
                    ha='center', va='center', transform=axes[1].transAxes, 
                    fontsize=14, fontweight='bold')
        axes[1].text(0.5, 0.5, f"Confidence: {result['confidence']:.3f}", 
                    ha='center', va='center', transform=axes[1].transAxes, 
                    fontsize=12)
        axes[1].text(0.5, 0.3, f"Model: {result['ai_insight']}", 
                    ha='center', va='center', transform=axes[1].transAxes, 
                    fontsize=10)
        axes[1].set_title('Classification Result')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        summary = f"""CT Image Analysis Complete:
- Classification: {result['label_name']}
- Confidence: {result['confidence']:.3f}
- Image dimensions: {img_array.shape}
- Model: Swin Transformer (Hugging Face)
- File format: Standard image"""
        
        metadata = {
            "file_type": "ct_image",
            "classification": result['label_name'],
            "confidence": float(result['confidence']),
            "dimensions": list(img_array.shape),
            "model": "swin_transformer"
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
            "summary": f"Error processing CT image: {str(e)}",
            "metadata": {"error": str(e)},
            "images": [],
            "figures": [],
            "audio": None
        }

def process_ct_unknown(file_path: str) -> Dict[str, Any]:
    """Handle unknown CT file types"""
    try:
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1]
        
        # Try to read as text to get preview
        try:
            with open(file_path, 'r') as f:
                content_preview = f.read(200)
        except:
            content_preview = "Binary file - content not displayable"
        
        summary = f"""CT File Analysis:
- File type: {file_ext if file_ext else 'Unknown'}
- File size: {file_size} bytes
- Supported formats: DICOM (.dcm), images (.jpg, .png, etc.)
- Current file format not directly supported by CT classifier
- Content preview: {content_preview[:100]}..."""
        
        metadata = {
            "file_type": "ct_unknown",
            "file_extension": file_ext,
            "file_size": file_size,
            "supported_formats": [".dcm", ".dicom", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "content_preview": content_preview[:100]
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
            "summary": f"Error analyzing CT file: {str(e)}",
            "metadata": {"error": str(e)},
            "images": [],
            "figures": [],
            "audio": None
        }
