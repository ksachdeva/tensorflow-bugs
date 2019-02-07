import tensorflow as tf
from tensorflow import keras


def build_simple_model():
    input_shape = (28, 28, 1)
    input_img = keras.layers.Input(shape=input_shape, name="input_img")
    x = keras.layers.Conv2D(32, kernel_size=(
        3, 3),  activation='relu')(input_img)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.models.Model(inputs=input_img, outputs=x)

    return model
