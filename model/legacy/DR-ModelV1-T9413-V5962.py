import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from tqdm.keras import TqdmCallback  # For custom progress bar (Optional)
import time
from tensorflow.keras.callbacks import Callback

# Custom callback to measure epoch time
class EpochTimeLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()  # Start time at the beginning of each epoch
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()  # End time at the end of each epoch
        epoch_duration = epoch_end_time - self.epoch_start_time  # Time taken for the epoch
        # Print the epoch duration and metrics at the end of each epoch
        print(f"Epoch {epoch + 1} - Time taken: {epoch_duration:.2f} seconds")
        print(f"Epoch {epoch + 1} - Training Loss: {logs['loss']:.4f}, Training Accuracy: {logs['accuracy']:.4f}")
        print(f"Epoch {epoch + 1} - Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")


# Local paths
data_dir = 'train_images'  
labels_path = 'train.csv'  

# Load dataset
data = pd.read_csv(labels_path)
data['id_code'] = data['id_code'] + '.png'

# Preprocess data
data['diagnosis'] = data['diagnosis'].astype(str)  # Required for flow_from_dataframe
train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['diagnosis'])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory=data_dir,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory=data_dir,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Build model
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(5, activation="softmax")(x)  # 5 classes for diabetic retinopathy severity

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model with verbose=3 and EpochTimeLogger callback
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,  # Training for 25 epochs
    verbose=3,  # Shows progress bar and accuracy/loss at each epoch
    callbacks=[
        TqdmCallback(),  # Optional: Custom progress bar
        EpochTimeLogger()  # Custom callback to log epoch efficiency
    ]
)

# Fine-tuning phase
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Fine-tuning training with verbose=3 and EpochTimeLogger callback
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,  # Training for 25 epochs
    verbose=3,  # Shows progress bar and accuracy/loss at each epoch
    callbacks=[
        TqdmCallback(),  # Optional: Custom progress bar
        EpochTimeLogger()  # Custom callback to log epoch efficiency
    ]
)

# Save model
model.save("DR-Model.h5")

# Evaluate model
val_preds = model.predict(val_generator)
val_preds_labels = np.argmax(val_preds, axis=1)
val_true_labels = val_generator.classes

kappa_score = cohen_kappa_score(val_true_labels, val_preds_labels, weights="quadratic")
print(f"Quadratic Weighted Kappa Score: {kappa_score}")
