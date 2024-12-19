import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import mlflow
import mlflow.tensorflow
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


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

        # Compute class weights for handling imbalance
        self.class_weights = self._compute_class_weights()

        mlflow.set_tracking_uri(uri="http://127.0.0.1:5500")
        self.experiment_name = config["mlflow"]["experiment_name"]
        mlflow.set_experiment(self.experiment_name)
        mlflow.tensorflow.autolog()

        self.callbacks = self._set_callbacks()

    def _compute_class_weights(self):
        """
        Compute class weights to handle class imbalance.
        """
        class_labels = self.train_generator.classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_labels),
            y=class_labels
        )
        return dict(enumerate(class_weights))

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
                    callbacks=self.callbacks,
                    class_weight=self.class_weights  # Use class weights
                )
            print(f"Training complete. Best model saved at {self.model_path}.")
            return history
        
        finally:
            mlflow.end_run()

    def evaluate(self):
        """
        Evaluate the model on the validation data generator with additional metrics.
        """
        try:
            from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

            print("Evaluating the model...")
            with mlflow.start_run(run_name="evaluation"):
                y_true = self.val_generator.classes
                y_pred_probs = self.model.predict(self.val_generator)
                y_pred = np.argmax(y_pred_probs, axis=1)

                # Compute additional metrics
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')

                results = self.model.evaluate(self.val_generator)
                mlflow.log_metrics({
                    "val_loss": results[0],
                    "val_accuracy": results[1],
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1_score": f1
                })

                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
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
