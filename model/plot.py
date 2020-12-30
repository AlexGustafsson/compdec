import os
import sys
from argparse import ArgumentParser

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

# It's usually not worth it to evalute these samples on the GPU - force use of the CPU instead
gpu_utilities.configure_gpu(use=False)

def plot_confusion_matrix(model_path, validation_set_path):
    """Create a confusion matrix plot for the validation set."""
    model = model_utilities.load_model(model_path)
    _, dataset_size, dataset_iterator = dataset_utilities.load_dataset(validation_set_path, epochs=1)

    required_batches = int(dataset_size / dataset_utilities.BATCH_SIZE)
    predictions = []
    correct = []
    for i in range(required_batches):
        samples, labels = next(dataset_iterator)
        batch_predictions = model.predict_on_batch(samples).tolist()
        predictions += [numpy.argmax(prediction, axis=0) for prediction in batch_predictions]
        correct += list(labels)
    predictions = numpy.array(predictions)
    correct = numpy.array(correct)
    confusion_matrix = tf.math.confusion_matrix(correct, predictions)
    plt.figure(figsize=(len(dataset_utilities.CLASS_NAMES), len(dataset_utilities.CLASS_NAMES)))
    seaborn.heatmap(confusion_matrix, square=True, xticklabels=dataset_utilities.CLASS_NAMES, cbar=False, yticklabels=dataset_utilities.CLASS_NAMES, annot=True, fmt='g', cmap=plt.get_cmap("Blues"))
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.yticks(rotation = 0)
    plt.show()

def plot_accuracy(model_path, validation_set_path):
    """Create a plot of prediction and confidence of some samples."""
    model = model_utilities.load_model(model_path)
    _, dataset_size, dataset_iterator = dataset_utilities.load_dataset(validation_set_path, epochs=1)

    samples, labels = next(dataset_iterator)

    predictions = model.predict_on_batch(samples)
    prediction_labels = ["{} ({:.4f})".format(dataset_utilities.CLASS_NAMES[numpy.argmax(prediction)], max(prediction)) for prediction in predictions]

    plt.figure(figsize=(10,10))
    # Batched dataset will return a series of samples
    for i in range(min(25, len(samples))):
        sample = samples[i]
        label = labels[i]
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample, cmap=plt.cm.binary)
        plt.xlabel("{} ({})".format(prediction_labels[i], dataset_utilities.CLASS_NAMES[label]))
    plt.show()

def plot_samples(validation_set_path):
    """Create a plot of some samples."""
    _, _, dataset_iterator = dataset_utilities.load_dataset(validation_set_path, epochs=1)

    samples, labels = next(dataset_iterator)

    plt.figure(figsize=(10,10))
    for i in range(min(25, len(samples))):
        sample = samples[i]
        label = labels[i]
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample, cmap=plt.cm.binary)
        plt.xlabel(dataset_utilities.CLASS_NAMES[label])
    plt.show()

def main():
    parser = ArgumentParser(description="A tool to create plots from a trained model")
    parser.add_argument("--type", required=True, type=str, choices=["confusion-matrix", "accuracy", "samples"], help="The type of plot to create")
    parser.add_argument("--model", required=False, type=str, help="Path to the model file")
    parser.add_argument("--strata", required=False, type=str, help="Path to the strata to use for data")
    options = parser.parse_args()

    if options.type == "confusion-matrix" and options.model and options.strata:
        plot_confusion_matrix(options.model, options.strata)
    elif options.type == "accuracy" and options.model and options.strata:
        plot_accuracy(options.model, options.strata)
    elif options.type == "samples" and options.strata:
        plot_samples(options.strata)
    else:
        parser.print_help()
        quit(1)

if __name__ == "__main__":
    main()
