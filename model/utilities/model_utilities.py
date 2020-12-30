import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy

sys.path.insert(0, os.path.abspath("./model/utilities"))
import dataset_utilities

def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)

def save_model(model, path):
    """Save the entire model."""
    model_directory = os.path.dirname(path)
    Path(model_directory).mkdir(parents=True, exist_ok=True)
    model.save(path)

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(dataset_utilities.IMAGE_SIZE, dataset_utilities.IMAGE_SIZE, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(len(dataset_utilities.CLASS_NAMES), activation="softmax"))
    return model

def initialize_model_from_cache(model, checkpoint_path, checkpoint_directory):
    """Loads weights from a stored checkpoint or initializes the model if necessary."""
    # NOTE: Assumes the checkpoint_path contains one named format item; {epoch}
    initial_epoch = 0
    saved_epochs = sorted([int(os.path.splitext(os.path.basename(file))[0]) for file in os.listdir(checkpoint_directory) if os.path.isfile(os.path.join(checkpoint_directory, file))], reverse=True)
    if len(saved_epochs) == 0:
        print("Storing fresh model")
        model.save_weights(checkpoint_path.format(epoch=0))
    else:
        initial_epoch = saved_epochs[0]
        checkpoint = checkpoint_path.format(epoch=initial_epoch)
        print("Loading the latest checkpoint: {} (epoch {})".format(checkpoint, initial_epoch))
        model.load_weights(checkpoint)
    return initial_epoch
