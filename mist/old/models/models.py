from tensorflow.keras import layers
from tensorflow.keras.models import Model

class UNet:

    def __init__(self, input_shape, init_filters, num_class, depth, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.init_filters = init_filters
        self.num_class = num_class
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Parameters for each keras layer that we use.
        # I like to keep them all in one place (i.e., a params dictionary)
        self.params = dict()
        self.params['conv'] = dict(kernel_size = 3, activation = 'relu', padding = 'same')
        self.params['maxpool2d'] = dict(pool_size = (2, 2), strides = (2, 2))
        self.params['maxpool3d'] = dict(pool_size = (2, 2, 2), strides = (2, 2, 2))
        self.params['point_conv'] = dict(kernel_size = 1, activation = 'relu', padding = 'same')
        self.params['trans_conv'] = dict(kernel_size = 2, strides = 2)

    def conv(self, x, filters):

        if len(self.input_shape) == 3:
            x = layers.Conv2D(filters = filters, **self.params['conv'])(x)
        else:
            x = layers.Conv3D(filters = filters, **self.params['conv'])(x)
        return x

    def block(self, x, filters):

        for _ in range(self.convs_per_block):
                x = self.conv(x, filters)

        return x

    def encoder(self, x):

        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            skips.append(self.block(x, filters))
            if len(self.input_shape) == 3:
                x = layers.MaxPooling2D(**self.params['maxpool2d'])(skips[i])
            else:
                x = layers.MaxPooling3D(**self.params['maxpool3d'])(skips[i])

        # Bottleneck
        skips.append(self.block(x, self.init_filters * (self.mul_on_downsample) ** self.depth))

        return skips

    def decoder(self, skips):

        x = skips[-1]
        skips = skips[:-1]
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            if len(self.input_shape) == 3:
                x = layers.Conv2DTranspose(filters, **self.params['trans_conv'])(x)
            else:
                x = layers.Conv3DTranspose(filters, **self.params['trans_conv'])(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)

        return x

    def softmax_output(self, x):

        if len(self.input_shape) == 3:
            x = layers.Conv2D(self.num_class, 1, activation = 'softmax')(x)
        else:
            x = layers.Conv3D(self.num_class, 1, activation = 'softmax')(x)

        return x

    def build_model(self):

        inputs = layers.Input(self.input_shape)

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])

        return model


class ResNet:

    def __init__(self, input_shape, init_filters, num_class, depth, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.init_filters = init_filters
        self.num_class = num_class
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Parameters for each keras layer that we use.
        # I like to keep them all in one place (i.e., a params dictionary)
        self.params = dict()
        self.params['conv'] = dict(kernel_size = 3, activation = 'relu', padding = 'same')
        self.params['maxpool2d'] = dict(pool_size = (2, 2), strides = (2, 2))
        self.params['maxpool3d'] = dict(pool_size = (2, 2, 2), strides = (2, 2, 2))
        self.params['point_conv'] = dict(kernel_size = 1, activation = 'relu', padding = 'same')
        self.params['trans_conv'] = dict(kernel_size = 2, strides = 2)

    def conv(self, x, filters):

        if len(self.input_shape) == 3:
            x = layers.Conv2D(filters = filters, **self.params['conv'])(x)
        else:
            x = layers.Conv3D(filters = filters, **self.params['conv'])(x)
        return x

    def block(self, x, filters):

        x = self.conv(x, filters)
        for _ in range(self.convs_per_block - 1):
            y = self.conv(x, filters)

        x = layers.Add()([x, y])

        return x

    def encoder(self, x):

        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            skips.append(self.block(x, filters))
            if len(self.input_shape) == 3:
                x = layers.MaxPooling2D(**self.params['maxpool2d'])(skips[i])
            else:
                x = layers.MaxPooling3D(**self.params['maxpool3d'])(skips[i])

        # Bottleneck
        skips.append(self.block(x, self.init_filters * (self.mul_on_downsample) ** self.depth))

        return skips

    def decoder(self, skips):

        x = skips[-1]
        skips = skips[:-1]
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            if len(self.input_shape) == 3:
                x = layers.Conv2DTranspose(filters, **self.params['trans_conv'])(x)
            else:
                x = layers.Conv3DTranspose(filters, **self.params['trans_conv'])(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)

        return x

    def softmax_output(self, x):

        if len(self.input_shape) == 3:
            x = layers.Conv2D(self.num_class, 1, activation = 'softmax')(x)
        else:
            x = layers.Conv3D(self.num_class, 1, activation = 'softmax')(x)

        return x

    def build_model(self):

        inputs = layers.Input(self.input_shape)

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])

        return model

class DenseNet:

    def __init__(self, input_shape, init_filters, num_class, depth, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.init_filters = init_filters
        self.num_class = num_class
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Parameters for each keras layer that we use.
        # I like to keep them all in one place (i.e., a params dictionary)
        self.params = dict()
        self.params['conv'] = dict(kernel_size = 3, activation = 'relu', padding = 'same')
        self.params['maxpool2d'] = dict(pool_size = (2, 2), strides = (2, 2))
        self.params['maxpool3d'] = dict(pool_size = (2, 2, 2), strides = (2, 2, 2))
        self.params['point_conv'] = dict(kernel_size = 1, activation = 'relu', padding = 'same')
        self.params['trans_conv'] = dict(kernel_size = 2, strides = 2)

    def conv(self, x, filters):

        if len(self.input_shape) == 3:
            x = layers.Conv2D(filters = filters, **self.params['conv'])(x)
        else:
            x = layers.Conv3D(filters = filters, **self.params['conv'])(x)
        return x

    def block(self, x, filters):

        for _ in range(self.convs_per_block):
            y = self.conv(x, filters)
            x = layers.concatenate([x, y])

        if len(x.input_shape) == 4:
            x = layers.Conv2D(filters, **self.params['point_conv'])(x)
        else:
            x = layers.Conv3D(filters, **self.params['point_conv'])(x)

        return x

    def encoder(self, x):

        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            skips.append(self.block(x, filters))
            if len(self.input_shape) == 3:
                x = layers.MaxPooling2D(**self.params['maxpool2d'])(skips[i])
            else:
                x = layers.MaxPooling3D(**self.params['maxpool3d'])(skips[i])

        # Bottleneck
        skips.append(self.block(x, self.init_filters * (self.mul_on_downsample) ** self.depth))

        return skips

    def decoder(self, skips):

        x = skips[-1]
        skips = skips[:-1]
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            if len(self.input_shape) == 3:
                x = layers.Conv2DTranspose(filters, **self.params['trans_conv'])(x)
            else:
                x = layers.Conv3DTranspose(filters, **self.params['trans_conv'])(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)

        return x

    def softmax_output(self, x):

        if len(self.input_shape) == 3:
            x = layers.Conv2D(self.num_class, 1, activation = 'softmax')(x)
        else:
            x = layers.Conv3D(self.num_class, 1, activation = 'softmax')(x)

        return x

    def build_model(self):

        inputs = layers.Input(self.input_shape)

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])

        return model