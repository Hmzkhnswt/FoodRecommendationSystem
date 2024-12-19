import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os
import mlflow
import mlflow.tensorflow
from datetime import datetime


class ModelTraining:
    def __init__(self, model, train_generator, val_generator, config):
        """
        Initialize the ModelTraining class.

        Args:
            model (tf.keras.Model): Compiled Keras model.
            train_generator (ImageDataGenerator): Training data generator.
            val_generator (ImageDataGenerator): Validation data generator.
            config (dict): Configurations loaded from params.yaml.
        """
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.config = config

        self.output_dir = config["paths"]["output_dir"]
        self.model_path = config["paths"]["best_model"]
        self.log_path = config["paths"]["training_log"]

        os.makedirs(self.output_dir, exist_ok=True)

        mlflow.set_tracking_uri(uri="http://127.0.0.1:5500")
        self.experiment_name = config["mlflow"]["experiment_name"]
        mlflow.set_experiment(self.experiment_name)
        mlflow.tensorflow.autolog()

        self.callbacks = self._set_callbacks()

    def _set_callbacks(self):
        """
        Set up Keras callbacks: ModelCheckpoint, EarlyStopping, and CSVLogger.
        """
        checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config["train"]["patience"],
            verbose=1,
            restore_best_weights=True
        )

        csv_logger = CSVLogger(self.log_path)

        return [checkpoint, early_stopping, csv_logger]

    def train(self):
        """
        Train the model using training and validation generators.
        """
        try:
            epochs = self.config["train"]["epochs"]
            print("Starting model training...")
            with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
                history = self.model.fit(
                    self.train_generator,
                    steps_per_epoch=self.train_generator.samples // self.config["train"]["batch_size"],
                    validation_data=self.val_generator,
                    validation_steps=self.val_generator.samples // self.config["train"]["batch_size"],
                    epochs=epochs,
                    callbacks=self.callbacks
                )
            print(f"Training complete. Best model saved at {self.model_path}.")
            return history
        
        finally:
            mlflow.end_run()

    def evaluate(self):
        """
        Evaluate the model on the validation data generator.
            """
        try:
            print("Evaluating the model...")
            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_params({
                    "learning_rate": 0.001,
                    "batch_size": self.config["train"]["batch_size"],
                    "epochs":  self.config["train"]["epochs"],
                    "optimizer": "adam"
                })
                results = self.model.evaluate(self.val_generator)
                mlflow.log_metrics({"val_loss": results[0], "val_accuracy": results[1]})
                print("Validation Results:", results)
            return results
        finally:
            mlflow.end_run()

    def save_model(self, save_path):
        """
        Save the final model.

        Args:
            save_path (str): Path to save the final model.
        """
        self.model.save(save_path)
        print(f"Final model saved at: {save_path}")
        mlflow.log_artifact(save_path)
