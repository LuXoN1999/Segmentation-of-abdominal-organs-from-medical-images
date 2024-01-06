from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, Activation, Concatenate, Input
from tensorflow.keras.models import Model


class Unet:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape

class UnetBuilder:
    @classmethod
    def conv_block(cls, inputs, num_filters):
        curr_layer = Conv2D(filters=num_filters, kernel_size=3, padding="same")(inputs)
        curr_layer = BatchNormalization()(curr_layer)
        curr_layer = Activation(activation="relu")(curr_layer)
        curr_layer = Conv2D(filters=num_filters, kernel_size=3, padding="same")(curr_layer)
        curr_layer = BatchNormalization()(curr_layer)
        curr_layer = Activation(activation="relu")(curr_layer)
        return curr_layer

    @classmethod
    def encoder_block(cls, inputs, num_filters):
        curr_layer = UnetBuilder.conv_block(inputs=inputs, num_filters=num_filters)
        pooling_layer = MaxPool2D(pool_size=(2, 2))(curr_layer)
        return curr_layer, pooling_layer

    @classmethod
    def decoder_block(cls, inputs, skip_conn_layer, num_filters):
        curr_layer = Conv2DTranspose(filters=num_filters, kernel_size=2, strides=2, padding="same")(inputs)
        curr_layer = Concatenate()([curr_layer, skip_conn_layer])
        curr_layer = UnetBuilder.conv_block(inputs=curr_layer, num_filters=num_filters)
        return curr_layer
