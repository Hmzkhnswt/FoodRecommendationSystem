import os
from FoodRecognition.components.dataingestion import DataIngestion
from FoodRecognition.constants.logging import logger
from FoodRecognition.constants.utils import read_yaml


def main():
    configuration_path = os.path.join("config/config.yaml")
    configs = read_yaml(configuration_path)

    kaggle_dataset = configs["data"]["kaggle_dataset"]
    artifacts_dir = configs["paths"]["artifacts_dir"]
    dataset_zip_name = configs["data"]["dataset_zip_name"]
    output_dir = configs["paths"]["extracted_data_dir"]

    logger.info(f"Kaggle dataset: {kaggle_dataset}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    logger.info(f"Dataset ZIP name: {dataset_zip_name}")
    logger.info(f"Output directory: {output_dir}")

    data_ingestion = DataIngestion(
        kaggle_dataset=kaggle_dataset,
        artifacts_dir=artifacts_dir,
        dataset_zip_name=dataset_zip_name,
        output_dir=output_dir
    )

    dataset_zip_path = data_ingestion.download_dataset()
    if dataset_zip_path:
        data_ingestion.extract_dataset()


if __name__ == "__main__":
    main()
