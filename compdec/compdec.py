import argparse
import inspect
import os
import sys
import hashlib
from argparse import ArgumentParser

# These are configurable, but highly dependant on the shipped model
CLASS_NAMES = ["7z", "brotli", "bzip2", "compress", "gzip", "lz4", "rar", "zip"]
CHUNK_SIZE = 4096
IMAGE_SIZE = 64

# Disable GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable verbose Tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def print_help(parser: ArgumentParser) -> None:
    formatted_class_names = "".join([class_name.rjust(int(80 / len(CLASS_NAMES))) for class_name in CLASS_NAMES])
    description = inspect.cleandoc("""
    CompDec is a tool to detect the compression algorithm used for a given file. It
    supports files of at least 4096 bytes and compressed files created using the
    following tools:

    {}

    The code is open source and freely available on GitHub:
    https://github.com/AlexGustafsson/compdec

    For instructions on how to train your own model, see the project's page on GitHub
    """.format(formatted_class_names))

    parser.print_help()
    print()
    print(description)

def print_version(model_path) -> None:
    hash = hashlib.sha256()
    with open(model_path, "rb") as file:
        block = file.read(4096)
        while len(block) > 0:
            hash.update(block)
            block = file.read(4096)
    print("CompDec v1.0.0")
    print("Model hash: {}".format(hash.hexdigest()))

def load_samples_from_file(sample_path):
    import numpy

    with open(sample_path, "rb") as sample_file:
        sample_file.seek(0, 2)
        file_size = sample_file.tell()
        sample_file.seek(0)

        chunks = int(file_size / CHUNK_SIZE)
        samples = []
        for i in range(0, chunks):
            sample = numpy.frombuffer(sample_file.read(CHUNK_SIZE), dtype=numpy.uint8) / 255.0
            if sample.shape[0] != CHUNK_SIZE:
                print("Warning: Skipping sample with incorrect size. Expected {}B got {}B".format(CHUNK_SIZE, sample.shape[0]), file=sys.stderr)
                continue
            # Make the sample is square (just like a 2D image)
            sample.shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
            samples.append(sample)
        return samples

def predict(sample_paths, model_path):
    import tensorflow
    import numpy

    model = tensorflow.keras.models.load_model(model_path)
    for sample_path in sample_paths:
        samples = load_samples_from_file(sample_path)

        if len(samples) == 0:
            print("There are no chunks big enough in the sample file. Expected at least {}B".format(CHUNK_SIZE))
            quit(1)

        def softmax(predictions):
            exponential_sum = numpy.exp(prediction_sum - numpy.max(prediction_sum))
            return exponential_sum / exponential_sum.sum(axis=0)

        predictions = model.predict(numpy.array(samples))
        prediction_sum = sum(predictions)
        normalized_predictions = softmax(prediction_sum)

        print(sample_path)
        for i in range(len(CLASS_NAMES)):
            print("{:9}: {:2.2f}%".format(CLASS_NAMES[i], normalized_predictions[i] * 100))

def assert_files(paths):
    files_exist = True
    for path in paths:
        if not os.path.exists(path):
            print("Error: File does not exist:", path)
            files_exist = False
    if not files_exist:
        quit(1)

def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("-v", "--version", action="store_true", help="print the program's version")
    parser.add_argument("-h", "--help", action="store_true", help="print this help message")
    parser.add_argument("--model", type=str, default="./compdec.h5", help="path to the pre-trained model to use")
    parser.add_argument("file", type=list, default=[], nargs="*")
    # TODO: add argument list for input files
    options = parser.parse_args()

    if options.help:
        print_help(parser)
    elif options.version:
        assert_files([options.model])
        print_version(options.model)
    elif len(options.file) > 0:
        paths = ["".join(file) for file in options.file]
        assert_files(paths)

        predict(paths, options.model)
    else:
        print_help(parser)


if __name__ == "__main__":
    main()
