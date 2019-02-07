import sys

import tensorflow as tf
from tensorflow import keras

from bug.model import build_simple_model
from bug.callback import ASimpleCallback
from bug.dataset import prepare_dataset


def main(args):
    dataset = prepare_dataset()

    # create our callbacks
    simple_callback = ASimpleCallback()

    # build the model architecture/layers
    model = build_simple_model()

    # use MirrorStrategy to make use of multiple GPUs
    distribution_strategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus=2)

    # distribution_strategy = None

    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy'], distribute=distribution_strategy)

    model.fit(dataset, steps_per_epoch=20,
              epochs=2, callbacks=[simple_callback], verbose=1)


if __name__ == '__main__':
    main(sys.argv)
