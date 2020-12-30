import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy
import matplotlib.pyplot as plt
import seaborn as seaborn

sys.path.insert(0, os.path.abspath("./model"))
from utilities import tensorflow_utilities

tensorflow_utilities.disable_verbose_logging()

import tensorflow as tf

from utilities import dataset_utilities
from utilities import model_utilities
from utilities import gpu_utilities

def train(options):
    gpu_utilities.configure_gpu(use=options.enable_gpu)

    ## Load the datasets
    training_dataset, training_dataset_size, _ = dataset_utilities.load_dataset(options.training_strata, epochs=options.epochs)
    evaluation_dataset, evaluation_dataset_size, _ = dataset_utilities.load_dataset(options.evaluation_strata, epochs=options.epochs)

    ## Create the model
    model = model_utilities.create_model()
    model.summary()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    callbacks = []

    ## Create a checkpoint callback
    checkpoint_path = "./data/checkpoints/{}/{}.hdf5".format(options.model_name, "{epoch:04d}")
    checkpoint_directory = os.path.dirname(checkpoint_path)
    # Create the checkpoints directory
    Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=int(training_dataset_size / dataset_utilities.BATCH_SIZE) * options.checkpoint_frequency
    )
    callbacks.append(checkpoint_callback)

    ## Optionally setup TensorBoard
    if options.enable_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./data/tensorboard", histogram_freq=1)
        callbacks.append(tensorboard_callback)

    ## Load the model from a previous checkpoint if there is any
    # Get the latest checkpoint (if any)
    initial_epoch = model_utilities.initialize_model_from_cache(model, checkpoint_path, checkpoint_directory)

    print("Training from epoch {}".format(initial_epoch))
    history = model.fit(
        training_dataset,
        epochs=initial_epoch + options.epochs,
        steps_per_epoch=int(training_dataset_size / dataset_utilities.BATCH_SIZE),
        validation_data=evaluation_dataset,
        validation_steps=int(evaluation_dataset_size / dataset_utilities.BATCH_SIZE),
        callbacks=callbacks,
        initial_epoch=initial_epoch
    )

    # Save the final training point
    model.save_weights(checkpoint_path.format(epoch=initial_epoch + options.epochs))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    if options.save_model:
        model_path = "./data/models/{}.h5".format(options.model_name)
        model_utilities.save_model(model, model_path)

def main():
    parser = ArgumentParser(description="A tool train a model")
    parser.add_argument("--model-name", required=True, type=str, help="Name of the model to create")
    parser.add_argument("--training-strata", required=True, type=str, help="Path to training strata")
    parser.add_argument("--evaluation-strata", required=True, type=str, help="Path to evaluation strata")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--checkpoint-frequency", default=5, type=int, help="The number of epochs to pass before saving a checkpoint")
    parser.add_argument("--save-model", default=False, action="store_true", help="Save the entire model when finished training")
    parser.add_argument("--enable-tensorboard", default=False, action="store_true", help="Enable TensorBoard")
    parser.add_argument("--enable-gpu", default=False, action="store_true", help="Whether or not to use GPU")
    options = parser.parse_args()

    train(options)

if __name__ == "__main__":
    main()
