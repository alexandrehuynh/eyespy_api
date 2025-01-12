import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("models/squat_model.h5")

def predict_squat(landmarks):
    """
    Predict squat type based on pose landmarks.
    Args:
        landmarks (list): List of 33 pose landmarks (each with x, y, z coordinates).
    Returns:
        str: Predicted squat type (e.g., 'Good squat', 'Bad squat', 'No squat').
    """
    # Prepare data for prediction
    landmarks = np.array([landmarks])  # Add batch dimension
    prediction = model.predict(landmarks)
    squat_type_index = np.argmax(prediction)
    return squat_type_index