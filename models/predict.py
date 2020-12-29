import sys
import os

# Disable GPU support for now
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy

# Workaround for 3080 - only works if there is a GPU attached
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

verbose = False

chunk_size = 4096
image_size = 64

class_names = ["7z", "brotli", "bzip2", "compress", "gzip", "lz4", "rar", "zip"]

model_path = sys.argv[1]
sample_path = sys.argv[2]

model = tf.keras.models.load_model(model_path)

if verbose:
    model.summary()

samples = []

with open(sample_path, "rb") as sample_file:
    file_size = os.path.getsize(sample_path)
    chunks = int(file_size / chunk_size)
    for i in range(0, chunks):
        sample = numpy.frombuffer(sample_file.read(chunk_size), dtype=numpy.uint8) / 255.0
        if sample.shape[0] != chunk_size:
            print("Warning: Skipping sample with incorrect size. Expected {}B got {}B".format(chunk_size, sample.shape[0]))
            continue
        # Make the sample is square (just like a 2D image)
        sample.shape = (image_size, image_size, 1)
        samples.append(sample)

if len(samples) == 0:
    print("There are no chunks big enough in the sample file. Expected at least {}B".format(chunk_size))
    quit(1)

if verbose:
    print("Evaluating {} samples".format(len(samples)))

def softmax(predictions):
    exponential_sum = numpy.exp(prediction_sum - numpy.max(prediction_sum))
    return exponential_sum / exponential_sum.sum(axis=0)

predictions = model.predict(numpy.array(samples))
prediction_sum = sum(predictions)
normalized_predictions = softmax(prediction_sum)

for i in range(len(class_names)):
    print("{:9}: {:2.2f}%".format(class_names[i], normalized_predictions[i] * 100))
