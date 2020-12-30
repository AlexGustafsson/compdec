import os

import tensorflow as tf

def configure_gpu(use=True):
    if use:
        # Workaround for RTX 30x
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
