import os
import numpy as np
import yaml
from keras.models import load_model
from sklearn.metrics import classification_report
import wandb


class ModelEvaluation:
    def __init__(self, model_path, test_generator):
        """
        Initialize the ModelEvaluation class.
        :param model_path: Path to the trained model file.
        :param test_generator: Keras data generator for the test dataset.
        """
        self.model_path = model_path
        self.test_generator = test_generator
        self.model = None

    def load_trained_model(self):
        """
        Load the trained model from the specified path.
        """
        print(f"Loading trained model from {self.model_path}...")
        self.model = load_model(self.model_path)
        print("Model loaded successfully.")

    def evaluate_model(self):
        """
        Evaluate the model using the test dataset.
        :return: A dictionary containing loss and accuracy.
        """
        print("Evaluating the model...")
        results = self.model.evaluate(self.test_generator, verbose=1)
        metrics = {
            "loss_main": results[0],
            "loss_aux": results[1],
            "accuracy_main": results[2],
            "accuracy_aux": results[3]
        }
        print(f"Evaluation Results: {metrics}")
        return metrics

    def get_classification_report(self):
        """
        Generate a classification report.
        :return: A classification report string.
        """
        print("Generating classification report...")
        y_true = self.test_generator.classes
        y_pred = self.model.predict(self.test_generator)
        y_pred_main = np.argmax(y_pred[0], axis=1)  # Predictions from the main output

        report = classification_report(
            y_true, y_pred_main,
            target_names=list(self.test_generator.class_indices.keys())
        )
        print(report)
        return report

    def log_to_wandb(self, metrics, classification_report):
        """
        Log evaluation metrics and classification report to W&B.
        :param metrics: Dictionary containing loss and accuracy.
        :param classification_report: Classification report string.
        """
        print("Logging results to Weights & Biases...")
        wandb.log(metrics)
        wandb.run.summary["classification_report"] = classification_report
        wandb.finish()
        print("Results logged to Weights & Biases.")
