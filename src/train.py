import tensorflow as tf
import os
from datetime import datetime
import time

# --- CONFIGURATION (Must match api.py and Docker setup) ---
MODEL_PATH = "models/crop_weed_model.h5" 
TRAIN_DIR = "data/train" 
IMAGE_SIZE = (150, 150)
EPOCHS = 3

# Global lock/status (To be handled by API, but added here for safety)
is_retraining = False

def retrain_pipeline():
    """
    Implements Trigger 3: Retraining.
    Loads existing model, loads data from TRAIN_DIR, retrains, and saves.
    """
    global is_retraining
    if is_retraining:
        print(f"[{datetime.now()}] WARNING: Retraining already in progress. Aborting.")
        return {"status": "warning", "message": "Retraining already in progress."}

    is_retraining = True
    print(f"[{datetime.now()}] --- RETRAINING STARTED (3 Epochs) ---")
    
    try:
        # 1. Load Model (The Custom Pre-Trained Model)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Base model not found. Upload a base model first.")
        
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[{datetime.now()}] Model loaded successfully for fine-tuning.")

        # 2. Load Data (Keras utility handles reading from subdirectories like 'crop' and 'weed')
        train_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR, 
            image_size=IMAGE_SIZE,
            batch_size=32,
            seed=123,
            shuffle=True
        )
        
        if len(train_ds) == 0:
            return {"status": "error", "message": "Training dataset is empty or invalid."}
        
        # 3. Data Preprocessing (Normalization) and Retrain
        # Normalization layer is applied to the dataset
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Re-compile model (important if optimizer/metrics need to change, though optional for fine-tuning)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print(f"[{datetime.now()}] Starting model fitting...")
        history = model.fit(train_ds, epochs=EPOCHS, verbose=1)
        
        # 4. Save Updated Model
        model.save(MODEL_PATH)
        print(f"[{datetime.now()}] --- Retraining finished. Model saved ---")
        
        return {
            "status": "success", 
            "final_accuracy": float(history.history['accuracy'][-1]),
            "epochs_run": EPOCHS
        }

    except FileNotFoundError as e:
        print(f"[{datetime.now()}] ERROR: {e}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        print(f"[{datetime.now()}] FATAL ERROR during retraining: {e}")
        return {"status": "error", "message": f"Retraining failed due to: {e}"}
    finally:
        is_retraining = False