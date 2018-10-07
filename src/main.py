from yolo_v3 import YOLO
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '../data/dog.jpg', 'Input image')
tf.app.flags.DEFINE_string('output_img', 'out.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', '../data/coco.names', 'File with class names')
tf.app.flags.DEFINE_string('cfg_file', '../cfg/yolov3.cfg', 'Configuration file for neural network')
tf.app.flags.DEFINE_string('raw_weights_file', '../weights/yolov3.weights', 'Binary weights file')
tf.app.flags.DEFINE_string('weights_file', '../weights/tensorflow/model.ckpt', 'Tensorflow weights file')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def main(argv=None):
    yolo = YOLO(FLAGS.class_names)
    yolo.create_model(FLAGS.cfg_file)
    # yolo.get_model_summary()
    # yolo.load_weights(FLAGS.raw_weights_file)
    yolo.load_weights(FLAGS.weights_file)
    yolo.predict(FLAGS.input_img, display=True)
    # yolo.save_weights(FLAGS.weights_file)


if __name__ == '__main__':
    tf.app.run()
