import tensorflow as tf

from models.layers import EncoderBlock, Bottleneck, DecoderBlock
from models.unet import UNetBlock

conv_kwargs = {"regularizer": "none",
               "norm": "batch",
               "activation": "relu",
               "down_type": "maxpool"}


class BaseModel(tf.keras.layers.Layer):

    def __init__(self,
                 block,
                 init_filters,
                 depth,
                 pocket,
                 global_depth,
                 **kwargs):
        super(BaseModel, self).__init__()

        # User defined inputs
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket
        self.global_depth = global_depth
        self.depth_offset = self.global_depth - self.depth

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        self.encoder = list()
        for i in range(self.depth):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.encoder.append(EncoderBlock(filters, block, **kwargs))

        filters = self.init_filters * self.mul_on_downsample ** self.depth
        self.bottleneck = Bottleneck(filters, block, **kwargs)

        self.decoder = list()
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.decoder.append(DecoderBlock(filters, block, **kwargs))

    def call(self, x, previous_skips):
        skips = list()
        new_skips = dict()
        for i, encoder_block in enumerate(self.encoder):
            skip, x = encoder_block(x)
            skips.append(skip)

            if i > 0:
                new_skips[str(self.depth_offset + i)] = skip

        x = self.bottleneck(x)

        skips.reverse()
        cnt = 0
        for skip, decoder_block in zip(skips, self.decoder):
            x = decoder_block(tf.keras.layers.concatenate([previous_skips[str(self.global_depth - 1 - cnt)], skip]), x)
            cnt += 1

        return x, new_skips


# Define modified unet class for fmgnet
class UNet(tf.keras.layers.Layer):

    def __init__(self,
                 init_filters,
                 depth,
                 pocket,
                 global_depth):
        super(UNet, self).__init__()

        self.base_model = BaseModel(UNetBlock,
                                    init_filters,
                                    depth,
                                    pocket,
                                    global_depth,
                                    **conv_kwargs)

    def call(self, x, previous_skips, **kwargs):
        return self.base_model(x, previous_skips, **kwargs)


"""
Define FMGNet class to implement FMG-Net.

This implementation constructs the FMG-Net to an arbitrary depth.
"""


class FMGNet(tf.keras.Model):

    def __init__(self,
                 n_classes,
                 init_filters,
                 depth,
                 pocket):
        super(FMGNet, self).__init__()

        # User defined inputs
        self.n_classes = n_classes
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket
        self.previous_skips = dict()

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        self.encoder = list()
        self.unets = list()
        for i in range(self.depth):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.encoder.append(EncoderBlock(filters, UNetBlock, **conv_kwargs))
            if i > 0:
                self.unets.append(UNet(filters, i, self.pocket, global_depth=self.depth))

        self.bottleneck = Bottleneck(self.init_filters * (self.mul_on_downsample ** self.depth),
                                     UNetBlock,
                                     **conv_kwargs)

        self.decoder = list()
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.decoder.append(DecoderBlock(filters, UNetBlock, **conv_kwargs))

        if not (self.n_classes is None):
            self.out = tf.keras.layers.Conv3D(self.n_classes, 1, dtype='float32')

    def call(self, x):
        for current_depth, encoder_block in enumerate(self.encoder):
            skip, x = encoder_block(x)
            self.previous_skips[str(current_depth)] = skip

        x = self.bottleneck(x)

        x = self.decoder[0](self.previous_skips[str(self.depth - 1)], x)
        current_depth = self.depth - 1
        for decoder, unet in zip(self.decoder[1:], self.unets):
            x, new_skips = unet(x,
                                previous_skips=self.previous_skips)

            # Don't update previous skips for last unet...
            # We don't use them later and don't need to waste memory doing that
            if current_depth > 1:
                for key in new_skips.keys():
                    self.previous_skips[key] = tf.keras.layers.concatenate([self.previous_skips[key], new_skips[key]])

            x = decoder(self.previous_skips[str(current_depth - 1)], x)

            current_depth -= 1

        if not (self.n_classes is None):
            x = self.out(x)

        self.previous_skips = dict()

        return x
