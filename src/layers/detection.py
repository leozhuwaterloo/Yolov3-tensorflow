import tensorflow as tf


def Detection(anchors, num_classes, img_size):
    """
    :param anchors: a list of anchors
    :param num_classes: number of classes
    :param img_size: the size of the image
    :return: detection results
    """
    return lambda x: detect(x, anchors, num_classes, img_size)


def detect(inputs, anchors, num_classes, img_size):
    """
    :param inputs: inputs
    :param anchors: a list of anchors
    :param num_classes: number of classes
    :param img_size: the size of the image
    :return: detection results
    """
    num_anchors = len(anchors)

    # Convert feature map to dimensions that we want
    bbox_attrs = 5 + num_classes

    shape = inputs.get_shape().as_list()
    grid_size = shape[1:3]
    dim = grid_size[0] * grid_size[1]

    # it is now [N, amount of bounding box, attribute of each bounding box]
    predictions = tf.reshape(inputs, [-1, num_anchors * dim, bbox_attrs])

    # Get relative anchor values
    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
    # Split the box attributes
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    # if grid_size is 8
    # x_y_offset is now [8 * 8, 2] = [64, 2]
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    # Duplicate x_y_offset anchor times so that it is now [64, 2 * num_anchor] = [64, 6]
    # Reshape it to [1, 192, 2]
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    # scale relative predicted box_centers offset to the actual coordinates
    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    # scale relative predicted box_sizes offset to the actual sizes
    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    # combine [each of the bounding box, class attributes]
    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)

    return detections_boxes(predictions)


def detections_boxes(detections):
    """
    :param detections: detection output with center x, center y, width and height
    :return: detection with coordinates converted to top left and bottom right points
    """
    center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2.0
    h2 = height / 2.0
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)
    return detections
