import os
import json
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from FoodRecognition.constants.logging import logger
import warnings

warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self, train_annotations, val_annotations, train_images, val_images, output_dir):
        """
        Initialize paths and output directories.
        """
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.train_images = train_images
        self.val_images = val_images
        self.output_dir = output_dir
        self.organized_train_dir = os.path.join(output_dir, "organized_training_data")
        self.organized_val_dir = os.path.join(output_dir, "organized_validation_data")
        
        os.makedirs(self.organized_train_dir, exist_ok=True)
        os.makedirs(self.organized_val_dir, exist_ok=True)

    def _load_annotations(self, annotations_file):
        """
        Load JSON annotation file.
        """
        try:
            with open(annotations_file, 'r') as file:
                data = json.load(file)
            logger.info(f"Loaded annotations from {annotations_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            raise
    
    def _process_data(self, images, categories, annotations):
        """
        Process and merge annotations into a DataFrame.
        """
        try:
            images_df = pd.DataFrame(images).rename(columns={'id': 'image_id'})[['image_id', 'file_name']]
            categories_df = pd.DataFrame(categories)[['id', 'name']].rename(columns={'id': 'category_id'})
            annotations_df = pd.DataFrame(annotations)[['image_id', 'category_id']]
            
            merged_df = annotations_df.merge(categories_df, on='category_id').merge(images_df, on='image_id')[
                ['file_name', 'name']
            ]
            logger.info("Successfully processed and merged data.")
            return merged_df
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def _organize_images(self, data_df, source_dir, target_dir, all_classes=None):
        """
        Organize images into category-specific directories. Ensures all classes exist.
        
        Args:
            data_df: DataFrame containing file_name and class name.
            source_dir: Path to the raw images directory.
            target_dir: Path to organize images into.
            all_classes: List of all class names (optional). Used to ensure consistency.
        """
        try:
            if all_classes is None:
                all_classes = data_df['name'].unique()
            for category in all_classes:
                category_dir = os.path.join(target_dir, category)
                os.makedirs(category_dir, exist_ok=True)

            for _, row in data_df.iterrows():
                src = os.path.join(source_dir, row['file_name'])
                dst = os.path.join(target_dir, row['name'], row['file_name'])
                if os.path.exists(src):
                    shutil.copy(src, dst)
            
            logger.info(f"Images organized into {target_dir}")
        except Exception as e:
            logger.error(f"Error organizing images: {e}")
            raise


    def process_training_data(self):
        """
        Process training data and organize into directories.
        """
        try:
            train_data = self._load_annotations(self.train_annotations)
            train_df = self._process_data(train_data['images'], train_data['categories'], train_data['annotations'])
            self.all_classes = train_df['name'].unique()  
            self._organize_images(train_df, self.train_images, self.organized_train_dir)
        except Exception as e:
            logger.error(f"Failed to process training data: {e}")

    def process_validation_data(self):
        """
        Process validation data and organize into directories.
        """
        try:
            val_data = self._load_annotations(self.val_annotations)
            val_df = self._process_data(val_data['images'], val_data['categories'], val_data['annotations'])
            self._organize_images(val_df, self.val_images, self.organized_val_dir, all_classes=self.all_classes)
        except Exception as e:
            logger.error(f"Failed to process validation data: {e}")

    def get_data_generators(self, target_size=(224, 224), batch_size=32):
        """
        Create data generators for training and validation.
        """
        try:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            val_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                self.organized_train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical'
            )

            val_generator = val_datagen.flow_from_directory(
                self.organized_val_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical'
            )

            logger.info("Data generators created successfully.")
            return train_generator, val_generator
        except Exception as e:
            logger.error(f"Error creating data generators: {e}")
            raise
