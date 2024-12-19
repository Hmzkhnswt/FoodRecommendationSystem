import os
import subprocess
from FoodRecognition.constants.logging import logger
from FoodRecognition.constants.utils import unzip_file


class DataIngestion:
    def __init__(self, kaggle_dataset, artifacts_dir, dataset_zip_name, output_dir):
        self.kaggle_dataset = kaggle_dataset
        self.artifacts_dir = artifacts_dir
        self.dataset_zip_path = os.path.join(self.artifacts_dir, dataset_zip_name)
        self.output_dir = output_dir

        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(os.getcwd())
        logger.info(f"Kaggle configuration directory set to: {os.environ['KAGGLE_CONFIG_DIR']}")

    def download_dataset(self):
        """
        Downloads the Kaggle dataset ZIP file using the Kaggle API.
        """
        if os.path.exists(self.dataset_zip_path):
            logger.info("Dataset ZIP file already exists. Skipping download.")
            return self.dataset_zip_path

        try:
            logger.info(f"Downloading dataset: {self.kaggle_dataset}")
            subprocess.run(
                [
                    "kaggle", "datasets", "download", "-d", self.kaggle_dataset,
                    "-p", self.artifacts_dir, "--force"
                ],
                check=True
            )
            logger.info(f"Dataset downloaded successfully: {self.dataset_zip_path}")
            return self.dataset_zip_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while downloading dataset: {e}")
            raise RuntimeError("Dataset download failed.") from e

    def extract_dataset(self):
        """
        Extracts the downloaded dataset ZIP file into the output directory.
        """
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            logger.info("Dataset is already extracted. Skipping extraction.")
            return

        if os.path.exists(self.dataset_zip_path):
            try:
                logger.info(f"Extracting dataset: {self.dataset_zip_path}")
                unzip_file(self.dataset_zip_path, self.output_dir)
                logger.info(f"Dataset extracted to {self.output_dir}")
            except Exception as e:
                logger.error(f"Error while extracting dataset: {e}")
                raise RuntimeError("Dataset extraction failed.") from e
        else:
            logger.warning("Dataset ZIP file not found. Cannot extract.")
