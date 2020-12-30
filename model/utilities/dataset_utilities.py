import csv
import os

import tensorflow as tf
import numpy

# Training / evaluation constants.
# These should be the same for training and evaluation
# for any given model
BATCH_SIZE = 64
CHUNK_SIZE = 4096
IMAGE_SIZE = 64
CLASS_NAMES = ["7z", "brotli", "bzip2", "compress", "gzip", "lz4", "rar", "zip"]

def load_samples_from_file(sample_path):
    with open(sample_path, "rb") as sample_file:
        sample_file.seek(0, 2)
        file_size = sample_file.tell()
        sample_file.seek(0)

        chunks = int(file_size / CHUNK_SIZE)
        samples = []
        for i in range(0, chunks):
            sample = numpy.frombuffer(sample_file.read(CHUNK_SIZE), dtype=numpy.uint8) / 255.0
            if sample.shape[0] != CHUNK_SIZE:
                print("Warning: Skipping sample with incorrect size. Expected {}B got {}B".format(CHUNK_SIZE, sample.shape[0]))
                continue
            # Make the sample is square (just like a 2D image)
            sample.shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
            samples.append(sample)
        return samples

def create_generator(strata_path: str):
    strata_file = open(strata_path, "r")
    reader = csv.reader(strata_file, delimiter=",", quotechar='"')
    def generator():
        # Reset the file (to support repeat)
        strata_file.seek(0)
        # Skip header
        next(reader, None)
        for file_path, offset, chunk_size, mime in reader:
            chunk_size = int(chunk_size)
            offset = int(offset)
            if chunk_size != CHUNK_SIZE:
                print("Warning: Skipping sample with incorrect chunk size. Expected {}".format(CHUNK_SIZE))
                continue

            with open(file_path, "rb") as sample_file:
                sample_file.seek(offset, 0)
                # Read the sample as a numpy array of bytes (uint8_t)
                sample = numpy.frombuffer(sample_file.read(chunk_size), dtype=numpy.uint8) / 255.0
                if sample.shape[0] != CHUNK_SIZE:
                    print("Warning: Skipping sample with incorrect size. Expected {}B got {}B".format(CHUNK_SIZE, sample.shape[0]))
                    continue
                # Make the sample square (just like a 2D image)
                sample.shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
                extension = os.path.splitext(file_path)[1].replace(".", "")
                # Use array to signal output classes for the input. Always one in this dataset
                output = CLASS_NAMES.index(extension)
                yield sample, output
    return generator

def create_dataset(generator, epochs, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.uint8),
        output_shapes=((64, 64, 1), ())
    ).repeat(epochs).batch(batch_size)

    return dataset

def get_dataset_size(strata_path: str):
    with open(strata_path, "r") as file:
        return sum(1 for line in file) - 1

def load_dataset(strata_path, epochs, batch_size=BATCH_SIZE):
    generator = create_generator(strata_path)
    dataset = create_dataset(generator, epochs, batch_size)
    iterator = dataset.__iter__()
    size = get_dataset_size(strata_path)
    return (dataset, size, iterator)
