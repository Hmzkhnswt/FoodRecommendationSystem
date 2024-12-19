import os
from FoodRecognition.components.dataprocessing import DataProcessor
from FoodRecognition.constants.logging import logger
from FoodRecognition.constants.utils import read_yaml

def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    try:
        config_path = "config/config.yaml"
        configs = read_yaml(config_path)

        train_annotations = os.path.join(configs["paths"]["processing_train_data"], "annotations.json")
        val_annotations = os.path.join(configs["paths"]["processing_val_data"], "annotations.json")
        train_images = os.path.join(configs["paths"]["processing_train_data"], "images")
        val_images = os.path.join(configs["paths"]["processing_val_data"], "images")
        output_dir = configs["paths"]["output_dir"]

        processor = DataProcessor(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            train_images=train_images,
            val_images=val_images,
            output_dir=output_dir
        )

        processor.process_training_data()
        processor.process_validation_data()

        batch_size = configs["train"]["batch_size"]
        train_generator, val_generator = processor.get_data_generators(batch_size=batch_size)

        logger.info("Data preprocessing pipeline completed successfully.")
        print("Data preprocessing complete. Check logs for details.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
