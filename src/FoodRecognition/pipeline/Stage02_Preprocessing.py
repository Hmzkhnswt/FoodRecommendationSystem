import os
import yaml
from FoodRecognition.components.dataprocessing import DataProcessor
from FoodRecognition.constants.logging import logger

# Function to load params.yaml
def load_params(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    try:
        params_path = "params.yaml"
        params = load_params(params_path)

        # Define paths from params.yaml
        train_annotations = os.path.join(params["paths"]["training_data"], "annotations.json")
        val_annotations = os.path.join(params["paths"]["validation_data"], "annotations.json")
        train_images = os.path.join(params["paths"]["training_data"], "images")
        val_images = os.path.join(params["paths"]["validation_data"], "images")
        output_dir = params["paths"]["output_dir"]

        # Initialize DataProcessor
        processor = DataProcessor(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            train_images=train_images,
            val_images=val_images,
            output_dir=output_dir
        )

        # Process training and validation data
        processor.process_training_data()
        processor.process_validation_data()

        # Generate data generators
        batch_size = params["train"]["batch_size"]
        train_generator, val_generator = processor.get_data_generators(batch_size=batch_size)

        logger.info("Data preprocessing pipeline completed successfully.")
        print("Data preprocessing complete. Check logs for details.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")

# Entry point
if __name__ == "__main__":
    main()
