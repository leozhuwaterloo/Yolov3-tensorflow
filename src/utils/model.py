from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, Lambda
from layers import Conv2D_BN_Leaky, Shortcut, Route, Detection, CacheResult, Summary


def create_model(blocks):
    """
    :param blocks: a list of blocks that describes the neural network
    :return: a neural network model
    """
    net_info = blocks[0]
    index = 0
    img_size = (int(net_info['width']), int(net_info['height']))
    model = Sequential()
    results = [None] * (len(blocks) - 1)
    yolo_indices = set()

    for x in blocks:
        if x["type"] == "net":
            continue

        if x['type'] == 'convolutional':
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            kernel_size = int(x['size'])
            strides = int(x['stride'])
            # pad = int(x['pad']) useless, instead we use strides to determine padding
            activation = x['activation']
            padding = 'valid' if strides > 1 else 'same'

            if batch_normalize and activation == 'leaky':
                layer = Lambda(Conv2D_BN_Leaky(filters, kernel_size, strides, padding,
                                               use_bias=bias), name='conv_bn_leaky_{0}'.format(index))
            else:
                layer = Conv2D(filters, kernel_size, strides, padding,
                               name='conv2d_{0}'.format(index))

        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            layer = UpSampling2D((stride, stride), name='upsample_{}'.format(index))
        elif x['type'] == 'shortcut':
            from_ = int(x["from"])
            layer = Lambda(Shortcut(from_, results, index), name='res_{0}'.format(index))
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]

            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            num_classes = int(x['classes'])

            yolo_indices.add(index)
            layer = Lambda(Detection(anchors, num_classes, img_size), name='yolo_{0}'.format(index))
        elif x['type'] == 'route':
            x['layers'] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            layer = Lambda(Route(start, end, results, index), name='route_{0}'.format(index))
        else:
            raise Exception(x['type'] + ' Unsupported !!!!!!!!!')

        model.add(Lambda(CacheResult(layer, results, index), name=layer.name))
        index += 1

    model.add(Lambda(Summary(results, yolo_indices), name='summary_{0}'.format(index)))
    return model
