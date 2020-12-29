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

print_samples = False

expected_chunk_size=4096
image_size=64
epochs = 10
batch_size = 64

def get_dataset_size(strata_path: str):
    with open(strata_path, "r") as file:
        return sum(1 for line in file) - 1

def create_generator(strata_path: str):
    strata_file = open(strata_path, "r")
    reader = csv.reader(strata_file, delimiter=",", quotechar='"')
    def generator():
        yielded = 0
        print("Resetting generator")
        # Reset the file (to support read)
        strata_file.seek(0)
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
                sample = numpy.frombuffer(sample_file.read(chunk_size), dtype=numpy.uint8) / 255.0
                if sample.shape[0] != expected_chunk_size:
                    print("Warning: Skipping sample with incorrect size. Expected {}B got {}B".format(expected_chunk_size, sample.shape[0]))
                    continue
                # Make the sample square (just like a 2D image)
                sample.shape = (image_size, image_size, 1)
                extension = os.path.splitext(file_path)[1].replace(".", "")
                # Use array to signal output classes for the input. Always one in this dataset
                output = class_names.index(extension)
                yield sample, output
                yielded += 1
        print("Exhausted dataset. Yielded {} items".format(yielded))
    return generator

def create_dataset(generator):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.uint8),
        output_shapes=((64, 64, 1), ())
    ).repeat(epochs).batch(batch_size)

    return dataset

train_dataset_size = get_dataset_size(sys.argv[1])
train_dataset = create_dataset(create_generator(sys.argv[1]))

test_dataset_size = get_dataset_size(sys.argv[2])
test_dataset = create_dataset(create_generator(sys.argv[2]))

# Show 10 samples
if print_samples:
    plt.figure(figsize=(10,10))
    dataset_iterator = train_dataset.__iter__()
    # Batched dataset will return a series of samples
    samples = next(dataset_iterator)
    for i in range(25):
        sample = samples[0][i]
        label = samples[1][i]
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
    plt.show()
    quit()

# https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
# batch = split into parts of x elements at a time
# dataset = tf.data.Dataset.from_generator(create_generator(sys.argv[1]), output_types=(tf.float64)).batch(batch_size)
# https://stackoverflow.com/questions/61249708/valueerror-no-gradients-provided-for-any-variable-tensorflow-2-0-keras

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names), activation="softmax"))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# tf.enable_eager_execution()

# fit(
#    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
#    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
#    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
#    validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
#    use_multiprocessing=False, **kwargs
#)

# history = model.fit(x=dataset, epochs=epochs, steps_per_epoch=num_batches)
# history = model.fit(create_generator(sys.argv[1]), epochs=epochs)
history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=int(train_dataset_size / batch_size),
        validation_data=test_dataset,
        validation_steps=int(test_dataset_size / batch_size),
    )

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print(test_acc)

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print(test_acc)
