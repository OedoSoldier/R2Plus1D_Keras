import math
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding3D, Conv3D, BatchNormalization, ReLU, Add, GlobalAveragePooling3D, Reshape, Dense
from tensorflow.keras.utils import plot_model
import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def SpatioTemporalConv(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True):
    """Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """
    # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)

    def f(inputs):
        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(
            math.floor(
                (kernel_size[0] *
                 kernel_size[1] *
                 kernel_size[2] *
                 in_channels *
                 out_channels) /
                (
                    kernel_size[1] *
                    kernel_size[2] *
                    in_channels +
                    kernel_size[0] *
                    out_channels)))

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        x = ZeroPadding3D(
            padding=spatial_padding,
            data_format='channels_first')(inputs)
        x = Conv3D(
            intermed_channels,
            spatial_kernel_size,
            strides=spatial_stride,
            padding='valid',
            data_format='channels_first',
            use_bias=bias)(x)
        x = BatchNormalization(axis=1)(x)
        x = ReLU()(x)

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        x = ZeroPadding3D(
            padding=temporal_padding,
            data_format='channels_first')(x)
        x = Conv3D(
            out_channels,
            temporal_kernel_size,
            strides=temporal_stride,
            padding='valid',
            data_format='channels_first',
            use_bias=bias)(x)

        return x
    return f


def SpatioTemporalResBlock(
        in_channels,
        out_channels,
        kernel_size,
        downsample=False):
    """Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
    """
    def f(inputs):

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.
        padding = kernel_size // 2

        if downsample:
            # downsample with stride = 2when producing the residual
            x = SpatioTemporalConv(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=2)(inputs)

            # downsample with stride =2 the input x
            inputs = SpatioTemporalConv(
                in_channels, out_channels, 1, stride=2)(inputs)
            inputs = BatchNormalization(axis=1)(inputs)
        else:
            x = SpatioTemporalConv(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding)(inputs)

        x = BatchNormalization(axis=1)(x)
        x = ReLU()(x)

        # standard conv->batchnorm->ReLU
        x = SpatioTemporalConv(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding)(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([inputs, x])
        x = ReLU()(x)
        return x
    return f


def SpatioTemporalResLayer(
        in_channels,
        out_channels,
        kernel_size,
        layer_size,
        block_type=SpatioTemporalResBlock,
        downsample=False):
    """Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
    """
    def f(inputs):
        # implement the first block
        x = block_type(
            in_channels,
            out_channels,
            kernel_size,
            downsample)(inputs)

        # prepare module list to hold all (layer_size - 1) blocks
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by
            # default
            x = block_type(out_channels, out_channels, kernel_size)(x)
        return x
    return f


def R2Plus1DNet(layer_sizes, block_type=SpatioTemporalResBlock):
    """Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """
    def f(inputs):

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        x = SpatioTemporalConv(
            3, 64, [
                3, 7, 7], stride=[
                1, 2, 2], padding=[
                1, 3, 3])(inputs)
        # output of conv2 is same size as of conv1, no downsampling needed.
        # kernel_size 3x3x3
        x = SpatioTemporalResLayer(
            64, 64, 3, layer_sizes[0], block_type=block_type)(x)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        x = SpatioTemporalResLayer(
            64,
            128,
            3,
            layer_sizes[1],
            block_type=block_type,
            downsample=True)(x)
        x = SpatioTemporalResLayer(
            128,
            256,
            3,
            layer_sizes[2],
            block_type=block_type,
            downsample=True)(x)
        x = SpatioTemporalResLayer(
            256,
            512,
            3,
            layer_sizes[3],
            block_type=block_type,
            downsample=True)(x)

        # global average pooling of the output
        x = GlobalAveragePooling3D(data_format='channels_first')(x)
        # x = Reshape((-1, 512))(x)
        return x
    return f


def R2Plus1DClassifier(
        num_classes,
        layer_sizes,
        block_type=SpatioTemporalResBlock):
    """Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """
    inputs = Input((32, 3, 112, 112))
    x = R2Plus1DNet(layer_sizes, block_type)(inputs)
    outputs = Dense(num_classes, activation='linear')(x)
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name='r2plus1d_18')
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


if __name__ == '__main__':
    model = R2Plus1DClassifier(num_classes=1, layer_sizes=[2, 2, 2, 2])
    print(model.summary())
    plot_model(model, to_file='model.png')
