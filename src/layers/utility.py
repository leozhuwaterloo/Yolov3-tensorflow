import tensorflow as tf


def CacheResult(func, results, index):
    """
    A layer that cache the result for shortcut and route layers
    :param func: function to apply to the input
    :param results: a list to store output results
    :param index: index to store the result
    :return: a cache result layer
    """
    def cache_result(x):
        res = func(x)
        results[index] = res
        return res
    return lambda x: cache_result(x)


def Summary(results, yolo_indices):
    """
    A layer that concatenates yolo predictions from different scale
    :param results: a list of previous outputs of layers
    :param yolo_indices: index of yolo layers
    :return: a summary layer
    """
    def summary():
        res = []
        for yolo_index in yolo_indices:
            res.append(results[yolo_index])
        return tf.concat(res, axis=1)
    return lambda _: summary()
