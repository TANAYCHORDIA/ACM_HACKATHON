import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess PIL Image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Target dimensions (height, width)
    
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize pixel values to [0, 1]
    image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array
