import os
import sys
import yaml
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Input, concatenate
)
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
sys.path.append("/Users/mac/Desktop/FoodRecommendation")
from src.FoodRecognition.components.modeltraining import ModelTraining

def load_params(config_path="params.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

CONFIG = load_params()

img_height, img_width = 224, 224
batch_size = CONFIG["train"]["batch_size"]
num_classes = 498

train_dir = CONFIG["paths"]["training_data"] 
val_dir = CONFIG["paths"]["validation_data"]

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

input_layer = Input(shape=(224, 224, 3))

# Initial convolution and pooling layers
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
x = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Inception module 3a
branch1_3a = Conv2D(64, (1, 1), padding='same', activation='relu')(x)

branch2_3a = Conv2D(96, (1, 1), padding='same', activation='relu')(x)
branch2_3a = Conv2D(128, (3, 3), padding='same', activation='relu')(branch2_3a)

branch3_3a = Conv2D(16, (1, 1), padding='same', activation='relu')(x)
branch3_3a = Conv2D(32, (5, 5), padding='same', activation='relu')(branch3_3a)

branch4_3a = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch4_3a = Conv2D(32, (1, 1), padding='same', activation='relu')(branch4_3a)

x = concatenate([branch1_3a, branch2_3a, branch3_3a, branch4_3a], axis=-1)

# Inception module 3b
branch1_3b = Conv2D(128, (1, 1), padding='same', activation='relu')(x)

branch2_3b = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
branch2_3b = Conv2D(192, (3, 3), padding='same', activation='relu')(branch2_3b)

branch3_3b = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
branch3_3b = Conv2D(96, (5, 5), padding='same', activation='relu')(branch3_3b)

branch4_3b = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch4_3b = Conv2D(64, (1, 1), padding='same', activation='relu')(branch4_3b)

x = concatenate([branch1_3b, branch2_3b, branch3_3b, branch4_3b], axis=-1)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Inception module 4a
branch1_4a = Conv2D(192, (1, 1), padding='same', activation='relu')(x)

branch2_4a = Conv2D(96, (1, 1), padding='same', activation='relu')(x)
branch2_4a = Conv2D(208, (3, 3), padding='same', activation='relu')(branch2_4a)

branch3_4a = Conv2D(16, (1, 1), padding='same', activation='relu')(x)
branch3_4a = Conv2D(48, (5, 5), padding='same', activation='relu')(branch3_4a)

branch4_4a = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch4_4a = Conv2D(64, (1, 1), padding='same', activation='relu')(branch4_4a)

x = concatenate([branch1_4a, branch2_4a, branch3_4a, branch4_4a], axis=-1)

# Auxiliary output
aux1 = AveragePooling2D((5, 5), strides=(3, 3))(x)
aux1 = Conv2D(128, (1, 1), padding='same', activation='relu')(aux1)
aux1 = Flatten()(aux1)
aux1 = Dense(1024, activation='relu')(aux1)
aux1 = Dropout(0.7)(aux1)
aux1 = Dense(498, activation='softmax')(aux1)  # Updated to match main output

# Inception module 4b
branch1_4b = Conv2D(160, (1, 1), padding='same', activation='relu')(x)

branch2_4b = Conv2D(112, (1, 1), padding='same', activation='relu')(x)
branch2_4b = Conv2D(224, (3, 3), padding='same', activation='relu')(branch2_4b)

branch3_4b = Conv2D(24, (1, 1), padding='same', activation='relu')(x)
branch3_4b = Conv2D(64, (5, 5), padding='same', activation='relu')(branch3_4b)

branch4_4b = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch4_4b = Conv2D(64, (1, 1), padding='same', activation='relu')(branch4_4b)

x = concatenate([branch1_4b, branch2_4b, branch3_4b, branch4_4b], axis=-1)

# Inception module 5a
branch1_5a = Conv2D(256, (1, 1), padding='same', activation='relu')(x)

branch2_5a = Conv2D(160, (1, 1), padding='same', activation='relu')(x)
branch2_5a = Conv2D(320, (3, 3), padding='same', activation='relu')(branch2_5a)

branch3_5a = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
branch3_5a = Conv2D(128, (5, 5), padding='same', activation='relu')(branch3_5a)

branch4_5a = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch4_5a = Conv2D(128, (1, 1), padding='same', activation='relu')(branch4_5a)

x = concatenate([branch1_5a, branch2_5a, branch3_5a, branch4_5a], axis=-1)

# Final layers
x = AveragePooling2D((7, 7))(x)
x = Dropout(0.4)(x)
x = Flatten()(x)
output = Dense(498, activation='softmax')(x)  # Main output

model = Model(inputs=input_layer, outputs=[output, aux1])

model.compile(
    optimizer="adam",
    loss=["categorical_crossentropy", "categorical_crossentropy"],
    metrics=["accuracy"]
)

trainer = ModelTraining(model, train_generator, val_generator, CONFIG)
trainer.train()
trainer.evaluate()
trainer.save_model(os.path.join(CONFIG["paths"]["output_dir"], "final_model.h5"))
