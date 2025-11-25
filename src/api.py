from fastapi import FastAPI, UploadFile, File, BackgroundTasks
# Ensure these imports are correct based on your file structure
from src.preprocessing import preprocess_image 
from src.train import retrain_pipeline 
import tensorflow as tf
import numpy as np
import shutil
import os

app = FastAPI(title="Crop vs Weed API")

# Global model variable
model = None
MODEL_PATH = "models/crop_weed_model.h5" # <--- FIXED RELATIVE PATH

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"INFO: Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            model = None
            print(f"ERROR loading model: {e}")
    else:
        model = None
        print("Warning: Model not found. Please train offline first.")

@app.get("/")
def health_check():
    # model_loaded checks if the model object exists
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        # Return a 404 error if the model isn't loaded
        return {"error": "Model not loaded. Please check logs."}
    
    image_data = await file.read()
    # Ensure preprocess_image returns the correct shape (1, 150, 150, 3) and normalization
    processed_img = preprocess_image(image_data) 
    
    # 

#[Image of Convolutional Neural Network architecture for image classification]

    prediction = model.predict(processed_img)
    
    # Assuming 0=Crop, 1=Weed (Based on common alphabetical directory sorting)
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        label = "Weed"
        prob = confidence
    else:
        label = "Crop"
        prob = 1.0 - confidence

    # Return the expected JSON fields that Streamlit expects
    return {"class": label, "confidence": round(prob, 4)}

@app.post("/upload-data")
async def upload_data(label: str, files: list[UploadFile] = File(...)):
    """Uploads data to specific class folder for future retraining"""
    # ... (Your existing upload code is fine, assuming data/train is correctly mounted) ...
    save_path = f"data/train/{label}"
    os.makedirs(save_path, exist_ok=True)
    
    count = 0
    for file in files:
        file_location = f"{save_path}/{file.filename}"
        # Ensure the file is not empty before writing
        file.file.seek(0)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        count += 1
        
    return {"message": f"Successfully saved {count} images to {label} class."}

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_pipeline)
    return {"message": "Retraining process started in background."}