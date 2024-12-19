import os
import sys
import yaml
import wandb
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from FoodRecognition.components.modelevaluation import ModelEvaluation

def load_config(config_path="params.yaml"):
    """
    Load the configuration from params.yaml.
    :param config_path: Path to the params.yaml file.
    :return: Dictionary containing configuration.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    config = load_config()

    wandb.init(
        project=config["wandb"]["project_name"],
        name=config["wandb"]["evaluation_run_name"]
    )

    model_path = config["paths"]["best_model"]
    test_dir = config["paths"]["test_data"]
    batch_size = config["evaluate"]["batch_size"]
    target_size = tuple(config["evaluate"]["image_size"])

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    evaluator = ModelEvaluation(model_path=model_path, test_generator=test_generator)

    evaluator.load_trained_model()

    metrics = evaluator.evaluate_model()

    report = evaluator.get_classification_report()

    metrics_path = config["paths"]["evaluation_metrics"]
    report_path = config["paths"]["classification_report"]

    with open(metrics_path, "w") as metrics_file:
        yaml.dump(metrics, metrics_file)

    with open(report_path, "w") as report_file:
        report_file.write(report)

    evaluator.log_to_wandb(metrics, report)

    print("Model evaluation completed successfully.")
