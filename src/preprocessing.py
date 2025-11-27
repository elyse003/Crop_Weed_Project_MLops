import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Configuration (must match model expectations)
IMAGE_SIZE = (150, 150)

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Implements Trigger 2: Data Preprocessing.
    Converts raw image bytes into a Keras-ready tensor: (1, 150, 150, 3).
    Includes resizing and normalization.
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # 1. Resize (to 150x150)
        image = image.resize(IMAGE_SIZE)
        
        # 2. Convert to NumPy array
        img_array = np.array(image, dtype=np.float32)
        
        # Handle grayscale images if they might appear (convert to RGB)
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        if img_array.shape[-1] == 4: # Handle RGBA
            img_array = img_array[..., :3]

        # 3. Normalize (Scale pixel values to 0-1)
        normalized_img = img_array / 255.0
        
        # 4. Add batch dimension (required by Keras model.predict)
        input_tensor = np.expand_dims(normalized_img, axis=0) 

        print(f"[PREPROC] Image successfully processed. Tensor shape: {input_tensor.shape}")
        return input_tensor
    
    except Exception as e:
        print(f"[PREPROC] Error during image preprocessing: {e}")
        # Return an empty array or raise a clear error to the API
        raise ValueError(f"Could not process image: {e}")