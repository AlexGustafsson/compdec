import os
import sys
from argparse import ArgumentParser

import numpy

sys.path.insert(0, os.path.abspath("./model"))
from utilities import tensorflow_utilities

tensorflow_utilities.disable_verbose_logging()

import tensorflow as tf

from utilities import dataset_utilities
from utilities import model_utilities
from utilities import gpu_utilities

# It's usually not worth it to evalute these samples on the GPU - force use of the CPU instead
gpu_utilities.configure_gpu(use=False)


def predict(model_path, sample_path):
    model = model_utilities.load_model(model_path)

    samples = dataset_utilities.load_samples_from_file(sample_path)

    if len(samples) == 0:
        print("There are no chunks big enough in the sample file. Expected at least {}B".format(dataset_utilities.CHUNK_SIZE))
        quit(1)

    def softmax(predictions):
        exponential_sum = numpy.exp(prediction_sum - numpy.max(prediction_sum))
        return exponential_sum / exponential_sum.sum(axis=0)

    predictions = model.predict(numpy.array(samples))
    prediction_sum = sum(predictions)
    normalized_predictions = softmax(prediction_sum)

    for i in range(len(dataset_utilities.CLASS_NAMES)):
        print("{:9}: {:2.2f}%".format(dataset_utilities.CLASS_NAMES[i], normalized_predictions[i] * 100))

def main():
    parser = ArgumentParser(description="A tool to run inference on a sample file")
    parser.add_argument("--model", required=True, type=str, help="Path to the model file")
    parser.add_argument("--sample", required=True, type=str, help="Path to the file to identify")
    options = parser.parse_args()

    predict(options.model, options.sample)

if __name__ == "__main__":
    main()
