import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def prepare_dataset():

    (x_train, y_train), _ = mnist.load_data()

    x_train = x_train.astype(np.float32) / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.one_hot(y_train, 10)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(32)

    return dataset
