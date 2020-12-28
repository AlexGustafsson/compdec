import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Workaround for 3080 - only works if there is a GPU attached
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Disable GPU support for now
import os
import sys
import csv
import numpy
from math import floor, sqrt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class_names = ["7z", "brotli", "bzip2", "compress", "gzip", "lz4", "rar", "zip"]

expected_chunk_size=4096
image_size=64

# TODO: Streaming
def load_dataset(strata_path: str):
    # List of samples (list of 4096 bytes)
    samples = []
    # List of labels for the samples
    labels = []

    with open(strata_path, "r") as strata_file:
        reader = csv.reader(strata_file, delimiter=",", quotechar='"')
        # Skip header
        next(reader, None)
        for file_path, offset, chunk_size, mime in reader:
            chunk_size = int(chunk_size)
            offset = int(offset)
            if chunk_size != expected_chunk_size:
                print("Warning: Skipping sample with incorrect chunk size. Expected {}".format(expected_chunk_size))
                continue

            with open(file_path, "rb") as sample_file:
                sample_file.seek(offset, 0)
                # Read the sample as a numpy array of bytes (uint8_t)
                sample = numpy.frombuffer(sample_file.read(chunk_size), dtype=numpy.uint8)
                # Make the sample square (just like a 2D image) - assumes 2^n chunks
                sample.shape = (image_size, image_size, 1)
                samples.append(sample)
                extension = os.path.splitext(file_path)[1].replace(".", "")
                # Use array to signal output classes for the input. Always one in this dataset
                labels.append([class_names.index(extension)])

    return (numpy.array(samples), numpy.array(labels))

def split_dataset(dataset, training_percentage):
    samples, labels = dataset
    training_sample_size = floor(len(samples) * training_percentage)
    training_samples = (samples[0:training_sample_size], labels[0:training_sample_size])
    test_samples = (samples[training_sample_size:], labels[training_sample_size:])
    return (training_samples, test_samples)

(train_images, train_labels), (test_images, test_labels) = split_dataset(load_dataset(sys.argv[1]), 0.6)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Show 10 samples
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(", ".join([class_names[x] for x in train_labels[i]]))
plt.show()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names)))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
