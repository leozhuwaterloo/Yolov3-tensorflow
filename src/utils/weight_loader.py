import numpy as np
import tensorflow as tf


def load_weights(var_list, weights_file):
    """
    Similar to https://github.com/mystic123/tensorflow-yolo-v3/blob/master/yolo_v3.py with a few tweaks
    :param var_list: a list of Tensorflow variables
    :param weights_file: the name of the weight file
    :return: a list of ops that can be used to set weights
    """
    # obtained the weights as a np array
    with open(weights_file, 'rb') as f:
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    i = 0
    load_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        print(var1.name)
        if 'conv2d' in var1.name.split('/')[-2]:
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1: i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    load_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                load_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            else:
                raise Exception("Unknown layer_2 name: {0}".format(var2.name))
                return None

            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            load_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
        else:
            raise Exception("Unknown layer_1 name: {0}".format(var1.name))

    return load_ops
