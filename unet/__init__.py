from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, Activation, Concatenate, Input, Layer
from tensorflow.keras.models import Model


class Unet:
    @classmethod
    def build_unet(cls, input_shape: tuple[int, int, int], num_classes: int, summarize: bool = False) -> Model:
        inputs = Input(shape=input_shape)
        encoder_block, conv_block, decoder_block = UnetBuilder.encoder_block, UnetBuilder.conv_block, UnetBuilder.decoder_block
        # encoder block
        s1, p1 = encoder_block(inputs=inputs, num_filters=64)
        s2, p2 = encoder_block(inputs=p1, num_filters=128)
        s3, p3 = encoder_block(inputs=p2, num_filters=256)
        s4, p4 = encoder_block(inputs=p3, num_filters=512)
        # bottleneck block
        b1 = conv_block(inputs=p4, num_filters=1024)
        # decoder block
        d1 = decoder_block(inputs=b1, skip_conn_layer=s4, num_filters=512)
        d2 = decoder_block(inputs=d1, skip_conn_layer=s3, num_filters=256)
        d3 = decoder_block(inputs=d2, skip_conn_layer=s2, num_filters=128)
        d4 = decoder_block(inputs=d3, skip_conn_layer=s1, num_filters=64)

        outputs = Conv2D(filters=num_classes, kernel_size=1, padding="same", activation="softmax")(d4)

        model = Model(inputs, outputs)
        if summarize:
            model.summary()
        return model


class UnetBuilder:
    @classmethod
    def conv_block(cls, inputs: Layer, num_filters: int) -> Layer:
        curr_layer = Conv2D(filters=num_filters, kernel_size=3, padding="same")(inputs)
        curr_layer = BatchNormalization()(curr_layer)
        curr_layer = Activation(activation="relu")(curr_layer)
        curr_layer = Conv2D(filters=num_filters, kernel_size=3, padding="same")(curr_layer)
        curr_layer = BatchNormalization()(curr_layer)
        curr_layer = Activation(activation="relu")(curr_layer)
        return curr_layer

    @classmethod
    def encoder_block(cls, inputs: Layer, num_filters: int) -> tuple[Layer, Layer]:
        curr_layer = UnetBuilder.conv_block(inputs=inputs, num_filters=num_filters)
        pooling_layer = MaxPool2D(pool_size=(2, 2))(curr_layer)
        return curr_layer, pooling_layer

    @classmethod
    def decoder_block(cls, inputs: Layer, skip_conn_layer: Layer, num_filters: int) -> Layer:
        curr_layer = Conv2DTranspose(filters=num_filters, kernel_size=2, strides=2, padding="same")(inputs)
        curr_layer = Concatenate()([curr_layer, skip_conn_layer])
        curr_layer = UnetBuilder.conv_block(inputs=curr_layer, num_filters=num_filters)
        return curr_layer
