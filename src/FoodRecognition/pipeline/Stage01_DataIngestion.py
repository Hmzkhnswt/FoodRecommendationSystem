import os
import sys
import yaml
sys.path.append("/Users/mac/Desktop/FoodRecommendation")
from src.FoodRecognition.components.dataingestion import DataIngestion
from src.FoodRecognition.constants.logging import logger
from src.FoodRecognition.constants.utils import read_yaml

def main():
    params_path = os.path.join("params.yaml")
    params = read_yaml(params_path)

    kaggle_url = params["data"]["kaggle_url"]
    artifacts_dir = params["paths"]["artifacts_dir"]
    dataset_zip_name = os.path.basename(params["data"]["dataset_zip_name"])
    output_dir = params["paths"]["extracted_data_dir"]

    logger.info(f"Using Kaggle URL: {kaggle_url}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    logger.info(f"Dataset ZIP name: {dataset_zip_name}")
    logger.info(f"Output directory: {output_dir}")

    data_ingestion = DataIngestion(
        kaggle_url=kaggle_url,
        artifacts_dir=artifacts_dir,
        dataset_zip_name=dataset_zip_name,
        output_dir=output_dir
    )

    dataset_zip_path = data_ingestion.download_dataset()

    if dataset_zip_path:
        data_ingestion.extract_dataset()

if __name__ == "__main__":
    main()
