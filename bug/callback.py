from tensorflow import keras
from tensorflow.keras.callbacks import Callback


class ASimpleCallback(Callback):

    def __init__(self):
        super(ASimpleCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        # in case of regular training
        # i.e. with out MirrorStrategy self.model
        # will of type keras.models.Model
        #
        # Whereas in the case of MirrorStrategy
        # it would of type DistributedModel
        print(self.model)

        # in case of MirrorStrategy the following will
        # fail
        print(
            f"Number of layers in distributed model are - {len(self.model.layers)}")

        # in case of MirrorStrategy following will
        # work however it is done by accessing a private property
        print(
            f"Number of layers in original model are - {len(self.model._original_model.layers)}")
