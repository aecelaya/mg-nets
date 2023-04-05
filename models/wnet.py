import tensorflow as tf

from models.layers import EncoderBlock, Bottleneck
from models.unet import UNetBlock

conv_kwargs = {"regularizer": "none",
               "norm": "batch",
               "activation": "relu",
               "down_type": "maxpool"}


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, block, **kwargs):
        super().__init__()
        self.trans_conv = tf.keras.layers.Conv3DTranspose(filters=filters,
                                                          kernel_size=2,
                                                          strides=2)
        self.block = block(filters, **kwargs)

    def call(self, skip, x):
        up = self.trans_conv(x)
        concat = tf.keras.layers.concatenate([*skip, up])
        out = self.block(concat)
        return out


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

        self.decoder = list()
        for i in range(self.depth, 0, -1):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.decoder.append(DecoderBlock(filters, block, **kwargs))

        self.encoder = list()
        for i in range(self.depth):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.encoder.append(EncoderBlock(filters, block, **kwargs))

        filters = self.init_filters * self.mul_on_downsample ** self.depth
        self.bottleneck = Bottleneck(filters, block, **kwargs)

        filters = self.init_filters * self.mul_on_downsample ** (self.depth - 1)
        self.subspike_decoder = DecoderBlock(filters, block, **kwargs)

        filters = self.init_filters * self.mul_on_downsample ** (self.depth - 1)
        self.subspike_encoder = EncoderBlock(filters, block, **kwargs)

        filters = self.init_filters * self.mul_on_downsample ** self.depth
        self.subspike_bottleneck = Bottleneck(filters, block, **kwargs)

    def call(self, x, previous_skips, previous_peaks):
        new_skips = dict()
        next_peaks = dict()
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block([*previous_skips[str(self.global_depth - 1 - i)],
                               *previous_peaks[str(self.global_depth - 1 - i)]],
                              x)

        for i, encoder_block in enumerate(self.encoder):
            skip, x = encoder_block(x)
            if i == 0:
                next_peaks[str(self.depth_offset + i)] = [skip]
            else:
                new_skips[str(self.depth_offset + i)] = [skip]

        x = self.bottleneck(x)

        x = self.subspike_decoder([*previous_skips[str(self.global_depth - 1)],
                                   *new_skips[str(self.global_depth - 1)]],
                                  x)

        skip, x = self.subspike_encoder(x)
        next_peaks[str(self.global_depth - 1)] = [skip]
        x = self.subspike_bottleneck(x)

        return x, new_skips, next_peaks


class SpikeNet(tf.keras.layers.Layer):

    def __init__(self,
                 init_filters,
                 depth,
                 pocket,
                 global_depth):
        super(SpikeNet, self).__init__()

        self.base_model = BaseModel(UNetBlock,
                                    init_filters,
                                    depth,
                                    pocket,
                                    global_depth,
                                    **conv_kwargs)

    def call(self, x, previous_skips, previous_peaks, **kwargs):
        return self.base_model(x, previous_skips, previous_peaks, **kwargs)


"""
Define WNet class to implement W-Net.

This implementation constructs the W-Net to an arbitrary depth.
"""


class WNet(tf.keras.Model):

    def __init__(self,
                 n_classes,
                 init_filters,
                 depth,
                 pocket):
        super(WNet, self).__init__()

        # User defined inputs
        self.n_classes = n_classes
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket
        self.previous_skips = dict()
        self.previous_peaks = dict()

        # Get number of spikes we use for the W-Net
        if self.depth == 3:
            self.spikes = [2]
        else:
            temp = [i for i in range(2, self.depth)] + [i for i in range(self.depth - 2, 1, -1)]
            self.spikes = [2] * (len(temp) * 2 - 1)
            self.spikes[0::2] = temp
            self.spikes.pop(0)
            self.spikes.pop(-1)

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Define main encoder branch
        self.encoder = list()
        for i in range(self.depth):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.encoder.append(EncoderBlock(filters, UNetBlock, **conv_kwargs))

        self.first_bottleneck = Bottleneck(self.init_filters * (self.mul_on_downsample ** self.depth),
                                           UNetBlock,
                                           **conv_kwargs)

        # Define first spike
        filters = self.init_filters * self.mul_on_downsample ** (self.depth - 1)
        self.subspike_decoder = DecoderBlock(filters, UNetBlock, **conv_kwargs)

        self.subspike_encoder = EncoderBlock(filters, UNetBlock, **conv_kwargs)

        filters = self.init_filters * self.mul_on_downsample ** self.depth
        self.subspike_bottleneck = Bottleneck(filters, UNetBlock, **conv_kwargs)

        # Define SpikeNets
        self.spikenets = list()
        for spike in self.spikes:
            self.spikenets.append(SpikeNet(init_filters=self.init_filters,
                                           depth=spike,
                                           pocket=self.pocket,
                                           global_depth=self.depth))

        self.decoder = list()
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.decoder.append(DecoderBlock(filters, UNetBlock, **conv_kwargs))

        self.out = tf.keras.layers.Conv3D(self.n_classes, 1, dtype='float32')

    def call(self, x):
        # Main encoder branch
        for current_depth, encoder_block in enumerate(self.encoder):
            skip, x = encoder_block(x)
            self.previous_skips[str(current_depth)] = [skip]
            self.previous_peaks[str(current_depth)] = []

        # First bottleneck
        x = self.first_bottleneck(x)

        # First subspike
        x = self.subspike_decoder(self.previous_skips[str(self.depth - 1)], x)
        skip, x = self.subspike_encoder(x)
        self.previous_peaks[str(self.depth - 1)] = [skip]
        x = self.subspike_bottleneck(x)

        # Apply all spikenets
        for spikenet in self.spikenets:
            x, new_skips, next_peaks = spikenet(x, self.previous_skips, self.previous_peaks)

            # Update skip connections
            for key in new_skips.keys():
                self.previous_skips[key].append(new_skips[key][0])

            # Update peaks
            for key in next_peaks.keys():
                self.previous_peaks[key] = next_peaks[key]

        # Final decoder branch
        current_depth = self.depth - 1
        for decoder in self.decoder:
            x = decoder([*self.previous_skips[str(current_depth)],
                         *self.previous_peaks[str(current_depth)]],
                        x)
            current_depth -= 1

        # Final output
        x = self.out(x)

        return x
