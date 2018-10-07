from utils import parse_cfg, create_model, load_weights, non_max_suppression, draw_boxes
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend as K


def need_model(func):
    def wrapper(*args, **kwargs):
        if not args[0].model:
            raise Exception("You must crate or load a model first!")
        return func(*args, **kwargs)

    return wrapper


def need_loaded(func):
    def wrapper(*args, **kwargs):
        if not args[0].loaded:
            raise Exception("You must load weights first!")
        return func(*args, **kwargs)
    return wrapper


class YOLO:
    def __init__(self, class_names):
        self.initialized = False
        self.loaded = False
        self.sess = None
        self.model = None
        self.classes = {}
        with open(class_names) as f:
            for i, name in enumerate(f):
                self.classes[i] = name

    def create_model(self, cfg_file):
        blocks = parse_cfg(cfg_file)
        self.model = create_model(blocks)
        self.initialized = False
        self.loaded = False
        print("Model created")

    @need_model
    def initialize_model(self):
        self.model.compile(loss='mse', optimizer='sgd')
        data = np.random.random((1, 416, 416, 3))
        labels = np.random.random((1, 10647, 85))
        with tf.variable_scope('detector'):
            self.model.fit(data, labels, epochs=0, batch_size=0)
        self.initialized = True
        print("Model initialized")

    @need_model
    def get_model_summary(self):
        if not self.initialized:
            self.initialize_model()
        self.model.summary()

    @need_model
    def load_weights(self, weights_file):
        if not self.initialized:
            self.initialize_model()
        if self.sess:
            self.sess.close()

        self.sess = tf.Session()
        if weights_file.endswith('.weights'):
            raw = True
        elif weights_file.endswith('.ckpt'):
            raw = False
        else:
            raise Exception("Unknown weight file type")

        if raw:
            var_list = tf.global_variables(scope='detector')
            load_ops = load_weights(var_list, weights_file)
            self.sess.run(load_ops)
        else:
            tf.train.Saver().restore(self.sess, weights_file)
        self.loaded = True
        print("Weights loaded")

    @need_model
    @need_loaded
    def save_weights(self, output_file):
        if self.sess:
            K.set_session(self.sess)
        tf.train.Saver().save(self.sess, output_file)
        print("Saved model to {0}".format(output_file))

    @need_model
    @need_loaded
    def predict(self, img, display=True, output_path='out.jpg'):
        img = Image.open(img)
        img_array = np.expand_dims(np.array(img.resize(size=(416, 416)), dtype=np.float32), 0) / 255.0
        if self.sess:
            K.set_session(self.sess)
        res = self.model.predict(img_array)
        print("Performing suppression")
        filtered_boxes = non_max_suppression(res, 0.25, 0.4)
        print(filtered_boxes)
        draw_boxes(filtered_boxes, img, self.classes, (416.0, 416.0))
        if display:
            img.show()
        img.save(output_path)