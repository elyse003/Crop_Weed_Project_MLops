import tensorflow as tf
import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes):
    """
    Converts raw bytes -> PIL Image -> Numpy Array -> Normalized Tensor
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0