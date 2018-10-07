from PIL import ImageDraw
import numpy as np


def draw_boxes(boxes, img, cls_names, detection_size):
    """
    :param boxes: a dictionary of {class: [prediction boxes]}
    :param img: original image
    :param cls_names: a dictionary of class names
    :param detection_size: the size of the detection
    :return:
    """
    draw = ImageDraw.Draw(img)
    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def convert_to_original_size(box, detection_size, original_size):
    """
    :param box: prediction box
    :param detection_size: size of the detection
    :param original_size: size of the original image
    :return: resized prediction boxed
    """
    ratio = original_size / detection_size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))
