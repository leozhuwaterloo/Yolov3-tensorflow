from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU
from utils import compose
import tensorflow as tf


def Conv2D_BN_Leaky(filters, kernel_size, strides, padding, **kwargs):
    """
    Convolutional Layer with padding, followed by batch normalization and leaky ReLU activation
    :param filters: filter for convolutional layer
    :param kernel_size: kernel_size for convolutional layer
    :param strides: strides for convolutional layer
    :param padding: padding for convolutional layer
    :param kwargs:
    :return: a composed desired layer
    """
    funcs = []
    if strides > 1:
        pad_total =  kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        funcs.append(ZeroPadding2D(((pad_beg, pad_end), (pad_beg, pad_end))))

    return compose(
        *funcs,
        Conv2D(filters, kernel_size, strides, padding, **kwargs),
        BatchNormalization(epsilon=1e-05, momentum=0.9),
        LeakyReLU(0.1)
    )


def Shortcut(relative_index, results, curr_index):
    """
    A layer that ADDS the result from previous layer and the layer indexed by (relative_index) from the current layer
    :param relative_index: a negative relative index
    :param results: a list of previous outputs of layers
    :param curr_index: current index
    :return: a shortcut layer
    """
    if relative_index >= 0:
        raise Exception("Shortcut layer must have negative relative index")
    return lambda x: x + results[curr_index + relative_index]


def Route(start, end, results, curr_index):
    """
    If end == 0: outputs the result of the layer indexed by (start) from the current layer
    If end > 0: outputs the result of the layer indexed by (start) from the current layer
        CONCATENATED with (end) indexed layer from start
    :param start: a negative relative index
    :param end: a positive absolute index
    :param results: a list of previous outputs of layers
    :param curr_index: current index
    :return: a route layer
    """
    if start >= 0 or end < 0:
        raise Exception("Route layer with start: {0}, end: {1} is not implemented".format(start, end))
    if end == 0:
        return lambda x: results[curr_index + start]
    else:
        return lambda x: tf.concat([results[curr_index + start], results[end]], axis=3)