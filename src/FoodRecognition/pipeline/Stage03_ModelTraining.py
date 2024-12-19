import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from FoodRecognition.components.modeltraining import ModelTraining
from FoodRecognition.constants.utils import read_yaml

config_path = "config/config.yaml"
configs = read_yaml(config_path)

img_height, img_width = 224, 224
batch_size = configs["train"]["batch_size"]
num_classes = 498

train_dir = configs["paths"]["training_data"]
val_dir = configs["paths"]["validation_data"]

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

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))

base_model.trainable = True
for layer in base_model.layers[:-30]: 
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Use SGD optimizer
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

trainer = ModelTraining(model, train_generator, val_generator, configs)
trainer.train()
trainer.evaluate()
trainer.save_model(os.path.join(configs["paths"]["output_dir"], "final_model_resnet50.h5"))
