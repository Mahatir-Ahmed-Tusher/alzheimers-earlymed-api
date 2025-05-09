import os
import numpy as np
import tensorflow as tf
from src.utils import preprocess_image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model from backend/Alzheimers/models/
model_path = os.path.join(os.path.dirname(__file__), "../models/alzheimer_cnn_model.h5")
try:
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise Exception(f"Error loading model: {str(e)}")

# Define class labels
class_labels = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

def run_inference(image):
    """
    Perform inference on the input image to detect Alzheimer's stage.
    """
    try:
        # Get model input shape
        input_shape = model.input_shape[1:3]  # e.g., (128, 128)
        expected_channels = model.input_shape[-1]  # e.g., 1 for grayscale, 3 for RGB

        # Preprocess the image
        image_array = preprocess_image(image, input_shape=input_shape, expected_channels=expected_channels)
        logger.info(f"Preprocessed image shape: {image_array.shape}")

        # Run inference
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions, axis=1)[0])

        # Format the result
        return {
            "prediction": class_labels[predicted_class_index],
            "confidence": confidence_score,
            "probabilities": {
                label: float(predictions[0][i]) for i, label in enumerate(class_labels)
            }
        }
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise Exception(f"Error during inference: {str(e)}")