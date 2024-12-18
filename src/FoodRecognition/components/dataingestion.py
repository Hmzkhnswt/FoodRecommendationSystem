import os
import subprocess
from FoodRecognition.constants.logging import logger
from FoodRecognition.constants.utils import unzip_file

class DataIngestion:
    def __init__(self, kaggle_url, artifacts_dir, dataset_zip_name, output_dir):
        """
        Initializes DataIngestion with the Kaggle dataset URL, artifacts directory, ZIP file name, and output directory.
        """
        self.kaggle_url = kaggle_url
        self.artifacts_dir = artifacts_dir
        self.dataset_zip = os.path.join(self.artifacts_dir, dataset_zip_name)
        self.output_dir = output_dir

        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)
            logger.info(f"Created artifacts directory: {self.artifacts_dir}")
        else:
            logger.info(f"Artifacts directory already exists: {self.artifacts_dir}")

    def download_dataset(self):
        """
        Downloads the Kaggle dataset ZIP file to the artifacts directory if it does not already exist.
        """
        if os.path.exists(self.dataset_zip):
            logger.info("Dataset ZIP file already exists. Skipping download.")
            return self.dataset_zip

        try:
            logger.info("Starting dataset download...")
            subprocess.run([
                "curl", "-L", "-o", self.dataset_zip, self.kaggle_url
            ], check=True)
            logger.info("Dataset downloaded successfully.")
            return self.dataset_zip
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while downloading dataset: {e}")
            return None

    def extract_dataset(self):
        """
        Extracts the downloaded dataset ZIP file into the output directory if not already extracted.
        """
        if os.path.exists(self.output_dir):
            logger.info("Dataset is already extracted. Skipping extraction.")
            return

        if os.path.exists(self.dataset_zip):
            try:
                logger.info("Starting extraction...")
                unzip_file(self.dataset_zip, self.output_dir)
                logger.info(f"Dataset extracted to {self.output_dir}")
            except Exception as e:
                logger.error(f"Error while extracting dataset: {e}")
        else:
            logger.warning("Dataset ZIP file not found. Extraction skipped.")
