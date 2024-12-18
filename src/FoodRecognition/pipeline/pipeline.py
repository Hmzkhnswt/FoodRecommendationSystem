import os
import sys
import logging
import yaml
sys.path.append("/Users/mac/Desktop/FoodRecommendation")
from src.FoodRecognition.constants.utils import read_yaml
from FoodRecognition.pipeline.Stage01_DataIngestion import main as data_ingestion_main
from FoodRecognition.pipeline.Stage02_Preprocessing import main as data_preprocessing_main
from FoodRecognition.pipeline.Stage03_ModelTraining import CONFIG as model_training_config
from FoodRecognition.pipeline.Stage04_ModelEvaluation import load_config as load_eval_config

params_path = os.path.join("params.yaml")

def configure_logging(log_dir="logs", log_filename="pipeline_execution.log"):
    """
    Set up logging configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, log_filename),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def main():
    """
    Main function to trigger the Food Recognition pipeline.
    """
    configure_logging()
    logging.info("Starting the Food Recognition pipeline...")

    try:
        params = read_yaml(params_path)
        logging.info("Parameters loaded successfully from params.yaml.")

        # Stage 01: Data Ingestion
        logging.info("Starting Stage 01: Data Ingestion...")
        data_ingestion_main()
        logging.info("Stage 01 completed successfully.\n")

        # Stage 02: Data Preprocessing
        logging.info("Starting Stage 02: Data Preprocessing...")
        data_preprocessing_main()
        logging.info("Stage 02 completed successfully.\n")

        # Stage 03: Model Training
        logging.info("Starting Stage 03: Model Training...")
        from FoodRecognition.pipeline.Stage03_ModelTraining import trainer
        trainer.train()
        trainer.evaluate()
        logging.info("Stage 03 completed successfully.\n")

        # Stage 04: Model Evaluation
        logging.info("Starting Stage 04: Model Evaluation...")
        from FoodRecognition.pipeline.Stage04_ModelEvaluation import evaluator
        eval_config = load_eval_config()
        evaluator.load_trained_model()
        metrics = evaluator.evaluate_model()
        report = evaluator.get_classification_report()

        metrics_path = eval_config["paths"]["evaluation_metrics"]
        report_path = eval_config["paths"]["classification_report"]

        # Save evaluation outputs
        with open(metrics_path, "w") as metrics_file:
            yaml.dump(metrics, metrics_file)
        with open(report_path, "w") as report_file:
            report_file.write(report)
        
        logging.info("Stage 04 completed successfully.\n")
        logging.info("Pipeline execution completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

