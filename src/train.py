import tensorflow as tf
import os

# --- FIX: USE RELATIVE PATHS FOR DOCKER ---
# These paths correspond to the volume mounts in docker-compose.yml
MODEL_PATH = "models/crop_weed_model.h5" 
TRAIN_DIR = "data/train" 
UPLOAD_DIR = "data/uploads" # Assuming uploads are placed here before merging/training

def retrain_pipeline():
    """
    Loads existing model, loads original data + uploaded data, retrains, and saves.
    """
    if not os.path.exists(MODEL_PATH):
        # This will now correctly check the container's /app/models folder
        return {"status": "error", "message": "Base model not found. Run notebook first."}

    # 1. Load Model
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Load Data (Keras utility assumes the TRAIN_DIR structure)
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR, # This now points to /app/data/train
            image_size=(150, 150),
            batch_size=32,
            seed=123
        )
    except Exception as e:
        # Catch if the directory is empty or malformed
        print(f"ERROR loading dataset: {e}")
        return {"status": "error", "message": "Error loading training data. Check data/train directory structure."}
    
    # Check if dataset is empty before training
    if len(train_ds) == 0:
        return {"status": "error", "message": "Training dataset is empty or invalid."}

    # 3. Normalization and Retrain
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, epochs=3)
    
    # 4. Save Updated Model
    model.save(MODEL_PATH)
    
    return {
        "status": "success", 
        "final_accuracy": float(history.history['accuracy'][-1]),
        "epochs_run": 3
    }