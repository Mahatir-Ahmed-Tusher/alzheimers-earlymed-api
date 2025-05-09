import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image: Image.Image, input_shape=(128, 128), expected_channels=1) -> np.ndarray:
    """
    Preprocess the input image for the Alzheimer's detection model.
    Args:
        image: PIL Image object
        input_shape: Tuple of (height, width) expected by the model
        expected_channels: Number of channels (1 for grayscale, 3 for RGB)
    Returns:
        Preprocessed image as a numpy array
    """
    try:
        # Convert image to RGB or Grayscale based on model expectation
        if expected_channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        # Resize image to the target size
        image = image.resize(input_shape)

        # Convert to array and normalize
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]

        # If grayscale, ensure the channel dimension is correct
        if expected_channels == 1 and len(image_array.shape) == 3 and image_array.shape[-1] != 1:
            image_array = np.mean(image_array, axis=-1, keepdims=True)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise Exception(f"Error preprocessing image: {str(e)}")