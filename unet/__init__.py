from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


class UnetBuilder:
    @classmethod
    def build_unet(cls, input_shape, n_classes):
        inputs = Input(input_shape)

        # Encoder path
        s1, p1 = cls.encoder_block(inputs=inputs, num_filters=64)
        s2, p2 = cls.encoder_block(inputs=p1, num_filters=128)
        s3, p3 = cls.encoder_block(inputs=p2, num_filters=256)
        s4, p4 = cls.encoder_block(inputs=p3, num_filters=512)

        # Bridge(bottleneck)
        b1 = cls.conv_block(inputs=p4, num_filters=1024)

        # Decoder path
        d1 = cls.decoder_block(inputs=b1, skip_conn_layer=s4, num_filters=512)
        d2 = cls.decoder_block(inputs=d1, skip_conn_layer=s3, num_filters=256)
        d3 = cls.decoder_block(inputs=d2, skip_conn_layer=s2, num_filters=128)
        d4 = cls.decoder_block(inputs=d3, skip_conn_layer=s1, num_filters=64)

        outputs = Conv2D(filters=n_classes, kernel_size=1, padding="same", activation="softmax")(d4)

        return Model(inputs, outputs, name="UNet")

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
        curr_layer = cls.conv_block(inputs=inputs, num_filters=num_filters)
        pooling_layer = MaxPool2D(pool_size=(2, 2))(curr_layer)
        return curr_layer, pooling_layer

    @classmethod
    def decoder_block(cls, inputs, skip_conn_layer, num_filters):
        curr_layer = Conv2DTranspose(filters=num_filters, kernel_size=(2, 2), strides=2, padding="same")(inputs)
        curr_layer = Concatenate()([curr_layer, skip_conn_layer])
        curr_layer = cls.conv_block(inputs=curr_layer, num_filters=num_filters)
        return curr_layer
