"""
FastAPI backend server for Footprint Direction Classification

Endpoints:
- POST /predict - Upload image and get prediction
- GET /metrics - Get model metrics
- GET /predictions - Get prediction history
- Static files served from /static/
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import json
import uuid
from datetime import datetime
from typing import List, Optional
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(title="Footprint Direction Classification API")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
MODEL_PATH = Path("runs/classify/train/weights/best.pt")
STATIC_DIR = Path("static")
OVERLAY_DIR = STATIC_DIR / "overlays"
PREDICTIONS_DB = Path("predictions.json")

# Create directories
STATIC_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(exist_ok=True, parents=True)

# Load model (lazy loading)
_model = None
_model_version = "YOLOv8n-cls"

def get_model():
    """Load and return the YOLOv8 model (singleton pattern)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        _model = YOLO(str(MODEL_PATH))
        # Debug: Print model class names
        if hasattr(_model, 'names'):
            print(f"Model class names: {_model.names}")
            print(f"Model names type: {type(_model.names)}")
    return _model

# Load prediction history
def load_predictions() -> List[dict]:
    """Load prediction history from JSON file."""
    if PREDICTIONS_DB.exists():
        try:
            with open(PREDICTIONS_DB, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_predictions(predictions: List[dict]):
    """Save prediction history to JSON file."""
    with open(PREDICTIONS_DB, 'w') as f:
        json.dump(predictions, f, indent=2)

def create_overlay_image(image_path: str, direction: str, confidence: float, all_probabilities: dict = None) -> str:
    """Create an overlay image with prediction text and all class probabilities."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        # Try with PIL if OpenCV fails
        pil_img = Image.open(image_path)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Resize if too large
    max_size = 800
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 8
    line_height = 25
    start_y = 15
    
    # Calculate text dimensions for all probabilities
    lines = []
    if all_probabilities:
        # Sort by probability (descending)
        sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            # Ensure class_name is a string (not a number)
            class_name_str = str(class_name)
            # Skip if it's still a number (shouldn't happen, but safety check)
            if class_name_str.isdigit():
                # Map numbers to directions if needed
                direction_map = {'0': 'North', '1': 'South', '2': 'East', '3': 'West'}
                class_name_str = direction_map.get(class_name_str, f'Class {class_name_str}')
            lines.append(f"{class_name_str}: {prob:.2f}")
    else:
        lines.append(f"{direction.upper()}: {confidence:.2f}")
    
    # Calculate box dimensions
    max_width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_width)
    
    box_width = max_width + padding * 2
    box_height = len(lines) * line_height + padding * 2
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (10, start_y), (10 + box_width, start_y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw text lines
    y_pos = start_y + padding + line_height
    for i, line in enumerate(lines):
        # Use green for the top prediction, white for others
        color = (0, 255, 0) if i == 0 else (255, 255, 255)
        cv2.putText(img, line, (10 + padding, y_pos), 
                    font, font_scale, color, thickness)
        y_pos += line_height
    
    # Save overlay
    overlay_filename = f"{uuid.uuid4().hex}.jpg"
    overlay_path = OVERLAY_DIR / overlay_filename
    cv2.imwrite(str(overlay_path), img)
    
    return f"/static/overlays/{overlay_filename}"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the direction of a footprint image."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        temp_path = STATIC_DIR / f"temp_{uuid.uuid4().hex}.{file.filename.split('.')[-1]}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Load model and predict
            model = get_model()
            results = model(str(temp_path), imgsz=224)
            probs = results[0].probs
            
            pred_idx = int(probs.top1)
            confidence = float(probs.top1conf)
            
            # Get class names from model - YOLOv8 stores them in model.names
            model = get_model()
            class_names = ['North', 'South', 'East', 'West']  # Default fallback
            
            # Try to get class names from the model
            if hasattr(model, 'names'):
                if isinstance(model.names, dict):
                    # If it's a dict like {0: 'North', 1: 'South', ...}
                    class_names = [model.names[i] for i in sorted(model.names.keys())]
                elif isinstance(model.names, list):
                    # If it's a list
                    class_names = list(model.names)
                elif hasattr(model.names, '__iter__'):
                    class_names = list(model.names)
            
            # Also try from results
            if hasattr(results[0], 'names'):
                result_names = results[0].names
                if isinstance(result_names, dict):
                    class_names = [result_names[i] for i in sorted(result_names.keys())]
                elif isinstance(result_names, list):
                    class_names = list(result_names)
            
            # Ensure we have valid class names (not numbers)
            if not class_names or any(isinstance(name, (int, float)) for name in class_names):
                class_names = ['North', 'South', 'East', 'West']
            
            # Get direction name
            if pred_idx < len(class_names):
                direction = str(class_names[pred_idx])
            else:
                direction = f'class_{pred_idx}'
            
            print(f"DEBUG: Class names extracted: {class_names}")
            print(f"DEBUG: Predicted index: {pred_idx}, Direction: {direction}")
            
            # Get all class probabilities
            all_probabilities = {}
            try:
                # YOLOv8 probs object - try multiple ways to access data
                prob_data = None
                
                # Method 1: Try probs.data (most common)
                if hasattr(probs, 'data'):
                    prob_data = probs.data
                    # Handle PyTorch tensors
                    if hasattr(prob_data, 'cpu'):
                        prob_data = prob_data.cpu().numpy()
                    elif hasattr(prob_data, 'numpy'):
                        prob_data = prob_data.numpy()
                    # Convert to list
                    if hasattr(prob_data, 'tolist'):
                        prob_data = prob_data.tolist()
                    elif hasattr(prob_data, '__iter__') and not isinstance(prob_data, str):
                        prob_data = list(prob_data)
                
                # Method 2: Try accessing directly as list/tuple
                if prob_data is None and hasattr(probs, '__iter__'):
                    try:
                        prob_data = list(probs)
                    except:
                        pass
                
                # Method 3: Try probs.top5conf or similar
                if prob_data is None:
                    # Get probabilities for each class by trying to access them
                    for i, class_name in enumerate(class_names):
                        # Ensure class_name is a string
                        class_name_str = str(class_name)
                        # If it's a number, map it to direction name
                        if class_name_str.isdigit():
                            direction_map = {0: 'North', 1: 'South', 2: 'East', 3: 'West'}
                            class_name_str = direction_map.get(int(class_name_str), f'Class {class_name_str}')
                        
                        try:
                            # Try to get probability for this class
                            if hasattr(probs, 'data') and len(probs.data) > i:
                                all_probabilities[class_name_str] = float(probs.data[i])
                            else:
                                all_probabilities[class_name_str] = 0.0
                        except:
                            all_probabilities[class_name_str] = 0.0
                else:
                    # Extract probabilities from prob_data
                    for i, class_name in enumerate(class_names):
                        # Ensure class_name is a string
                        class_name_str = str(class_name)
                        # If it's a number, map it to direction name
                        if class_name_str.isdigit():
                            direction_map = {0: 'North', 1: 'South', 2: 'East', 3: 'West'}
                            class_name_str = direction_map.get(int(class_name_str), f'Class {class_name_str}')
                        
                        if i < len(prob_data):
                            try:
                                all_probabilities[class_name_str] = float(prob_data[i])
                            except (ValueError, TypeError):
                                all_probabilities[class_name_str] = 0.0
                        else:
                            all_probabilities[class_name_str] = 0.0
                            
            except Exception as e:
                # If all else fails, create a simple dict with the predicted class
                print(f"Warning: Could not extract all probabilities: {e}")
                import traceback
                traceback.print_exc()
                # Use standard directions as fallback
                standard_directions = ['North', 'South', 'East', 'West']
                for name in standard_directions:
                    all_probabilities[name] = 1.0 if name == direction else 0.0
            
            # Ensure we have all classes with proper string names
            # Normalize class names to strings and ensure they're in the dict
            normalized_class_names = []
            for name in class_names:
                name_str = str(name)
                if name_str.isdigit():
                    direction_map = {0: 'North', 1: 'South', 2: 'East', 3: 'West'}
                    name_str = direction_map.get(int(name_str), f'Class {name_str}')
                normalized_class_names.append(name_str)
                if name_str not in all_probabilities:
                    all_probabilities[name_str] = 0.0
            
            # Also ensure we have the standard 4 directions
            standard_directions = ['North', 'South', 'East', 'West']
            for std_direction in standard_directions:
                if std_direction not in all_probabilities:
                    all_probabilities[std_direction] = 0.0
            
            # Align final direction/confidence with probability dictionary
            if all_probabilities:
                best_direction, best_prob = max(
                    all_probabilities.items(), key=lambda item: item[1]
                )
                direction = best_direction
                confidence = float(best_prob)
            
            # Debug: print probabilities
            print(f"All probabilities extracted: {all_probabilities}")
            print(f"Number of classes: {len(class_names)}, Probabilities: {len(all_probabilities)}")
            
            # Create overlay image with all probabilities
            overlay_path = create_overlay_image(str(temp_path), direction, confidence, all_probabilities)
            
            # Save prediction to history
            prediction = {
                "timestamp": datetime.now().isoformat(),
                "direction": direction,
                "confidence": confidence,
                "overlay": overlay_path,
                "image_filename": file.filename,
                "all_probabilities": all_probabilities,
            }
            predictions = load_predictions()
            predictions.insert(0, prediction)  # Add to beginning
            # Keep only last 100 predictions
            predictions = predictions[:100]
            save_predictions(predictions)
            
            response_data = {
                "direction": direction,
                "confidence": confidence,
                "overlay": overlay_path,
                "model_version": _model_version,
                "all_probabilities": all_probabilities,
            }
            print(f"Response data: {response_data}")  # Debug
            return JSONResponse(response_data)
        
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get model metrics from validation results."""
    try:
        # Read validation results CSV
        results_csv = Path("runs/classify/train/results.csv")
        if not results_csv.exists():
            raise HTTPException(status_code=404, detail="Training results not found")
        
        df = pd.read_csv(results_csv)
        
        # Get final metrics (last row)
        final_row = df.iloc[-1]
        final_accuracy = float(final_row.get('metrics/accuracy_top1', 0))
        
        # Try to read confusion matrix path (check both locations)
        confusion_matrix_path = None
        for path in [
            Path("runs/classify/val/confusion_matrix.png"),
            Path("runs/classify/train/confusion_matrix.png"),
        ]:
            if path.exists():
                confusion_matrix_path = path
                break
        
        confusion_matrix_image = None
        if confusion_matrix_path:
            # Copy to static directory
            static_cm_path = STATIC_DIR / "confusion_matrix.png"
            shutil.copy2(confusion_matrix_path, static_cm_path)
            confusion_matrix_image = "/static/confusion_matrix.png"
        
        # Get class names from model
        try:
            model = get_model()
            # Try to get class names from model
            class_names = ['North', 'South', 'East', 'West']  # Default
            # YOLOv8 classification models store class names in the model
            if hasattr(model, 'names'):
                class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
        except:
            class_names = ['North', 'South', 'East', 'West']
        
        return JSONResponse({
            "final": {
                "accuracy": final_accuracy,
            },
            "class_names": class_names,
            "confusion_matrix_image": confusion_matrix_image,
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")

@app.get("/predictions")
async def get_predictions(limit: int = 20):
    """Get prediction history."""
    try:
        predictions = load_predictions()
        return JSONResponse({
            "predictions": predictions[:limit]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Footprint Direction Classification API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

