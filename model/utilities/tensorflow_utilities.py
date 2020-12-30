import os

def disable_verbose_logging():
    """Don't let Tensorflow log debug / info messages. Should be done before importing Tensorflow."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
