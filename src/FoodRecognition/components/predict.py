import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import argparse

def load_params(config_path="params.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, target_size):
    try:
        # Load the image and resize it to the target size
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def predict_image(model, image_path, class_indices, target_size):
    """
    Run inference on a single image and return the predicted class.
    """
    # Preprocess the image
    image = load_and_preprocess_image(image_path, target_size)

    # Make predictions
    predictions = model.predict(image)
    main_output = predictions[0] if isinstance(predictions, list) else predictions  # Handle multi-output models
    predicted_class_index = np.argmax(main_output, axis=1)[0]
    
    # Map index to class label
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse the dictionary
    predicted_class = class_labels.get(predicted_class_index, "Unknown")

    return predicted_class, main_output[0][predicted_class_index]

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Food Recognition Inference Script")
    parser.add_argument("--image", required=True, help="Path to the input image for prediction.")
    args = parser.parse_args()

    # Load configuration
    params = load_params("params.yaml")
    model_path = params["paths"]["best_model"]
    target_size = tuple(params["evaluate"]["image_size"])

    # Load class indices (if available)
    class_indices_path = params["paths"]["class_indices"]
    if os.path.exists(class_indices_path):
        with open(class_indices_path, "r") as f:
            import json
            class_indices = json.load(f)
    else:
        print("Error: class_indices.json file not found.")
        sys.exit(1)

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        sys.exit(1)
    print("Loading model...")
    model = load_model(model_path)

    # Run inference
    print("Running prediction...")
    predicted_class, confidence = predict_image(model, args.image, class_indices, target_size)

    # Display results
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
