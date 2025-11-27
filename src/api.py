from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import List
# Ensure these imports are correct based on your file structure
from src.preprocessing import preprocess_image 
from src.train import retrain_pipeline 
import tensorflow as tf
import numpy as np
import os
import shutil
import io # Added for robust file reading

# --- CONFIGURATION ---
MODEL_PATH = "models/crop_weed_model.h5" 
TRAIN_DIR = "data/train" 
# Global model and status variables
model = None
retraining_in_progress = False

app = FastAPI(title="Crop vs Weed API")


@app.on_event("startup")
def load_model_on_startup():
    """Load the Keras model once when the application starts."""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # Use tf.keras.models.load_model which is thread-safe for reading
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            model = None
            print(f"[ERROR] loading model: {e}")
    else:
        model = None
        print("[WARNING] Model not found. Please ensure 'models/crop_weed_model.h5' exists.")

@app.get("/")
def health_check():
    """Health Check endpoint for API status and model availability."""
    return {"status": "online", "model_loaded": model is not None, "retraining": retraining_in_progress}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Inference endpoint using the loaded model."""
    if retraining_in_progress:
        raise HTTPException(status_code=503, detail="Model is currently being updated (retraining). Try again shortly.")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check backend logs.")
    
    # Read image data
    image_data = await file.read()
    
    # Trigger 2: Data Preprocessing
    # preprocess_image returns the (1, 150, 150, 3) tensor
    try:
        processed_img = preprocess_image(image_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Perform prediction
    prediction = model.predict(processed_img)
    
    # Assuming 0=Crop, 1=Weed (Based on Keras/alphabetical directory sorting)
    # The output is a single probability value (e.g., [0.98])
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        label = "weed"
        prob = confidence
    else:
        label = "crop"
        prob = 1.0 - confidence

    # Return the expected JSON fields that Streamlit expects
    return {"class": label, "confidence": round(prob, 4)}

@app.post("/upload-data")
async def upload_data(label: str, files: List[UploadFile] = File(...)):
    """
    Implements Trigger 1: Data File Uploading + Saving.
    Uploads data to specific class folder (data/train/[label]) for future retraining.
    """
    valid_label = label.lower()
    if valid_label not in ['crop', 'weed']:
        raise HTTPException(status_code=400, detail="Label must be 'crop' or 'weed'.")

    # Define the save path within the training data directory
    save_path = os.path.join(TRAIN_DIR, valid_label)
    os.makedirs(save_path, exist_ok=True)
    
    count = 0
    for file in files:
        file_location = os.path.join(save_path, file.filename)
        
        # File.file (a SpooledTemporaryFile) needs to be read and saved.
        # Ensure the cursor is at the beginning of the file stream
        file.file.seek(0)
        
        # Save the file content
        try:
            with open(file_location, "wb") as buffer:
                # Use io.BytesIO(await file.read()) if you need to read the content first, 
                # but shutil.copyfileobj is efficient for file-like objects.
                shutil.copyfileobj(file.file, buffer)
            count += 1
            print(f"[UPLOAD] Saved image to {file_location}")
        except Exception as e:
            print(f"[UPLOAD] Error saving {file.filename}: {e}")
            continue

    return {"message": f"Successfully saved {count} images to {valid_label} class. Retraining required."}


# --- Status Management for Background Task ---

def update_retraining_status(status: bool):
    """Simple synchronous function to update the global status."""
    global retraining_in_progress
    retraining_in_progress = status
    print(f"[STATUS] Retraining in progress set to: {status}")

def retraining_wrapper():
    """Wrapper function to handle status updates around the synchronous pipeline."""
    update_retraining_status(True)
    try:
        retrain_pipeline()
    finally:
        # Crucial: ensure status is reset even if training fails
        update_retraining_status(False)

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger 3: Retraining - Starts the synchronous pipeline in the background.
    """
    if retraining_in_progress:
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    
    # Add the wrapper function to the background tasks
    background_tasks.add_task(retraining_wrapper)
    
    return {"message": "Retraining process started in background. Monitor console for progress."}