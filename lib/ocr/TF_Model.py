import numpy as np
import tensorflow as tf

import os, sys
from skimage.transform import resize
from skimage.util import pad

from .utils.utils import resize_img, resize_batch

class Model():
    """Hair Model"""
    def __init__(self, graph_name, shape=(320, 320), device='/cpu:0'):
        # super(Model, self).__init__()

        self.graph_name  = graph_name
        self.device      = device
        self.graph       = self.build_model(graph_name, device)
        self.input_shape = shape


    def build_model(self, graph_name, device):
        with tf.device(device):
            output_graph_def = None
            with open(graph_name, 'rb') as f:
                output_graph_def = tf.GraphDef()
                output_graph_def.ParseFromString(f.read())
                
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(output_graph_def, name='')
                return graph


    def predict(self, img_batch, return_input=False):
        shape = self.input_shape
        if len(img_batch.shape) == 3:
            img_batch = resize_img(img_batch, shape=shape)
            img_batch = np.expand_dims(img_batch, 0)
        elif len(img_batch.shape) == 4:
            img_batch = resize_batch(img_batch, shape=shape)
            # for image_idx in range(img_batch.shape[0]):
            #     img_batch[image_idx] = resize_batch(img_batch[image_idx], shape=shape)
        elif len(img_batch.shape) == 2:
            img_batch = np.expand_dims(img_batch, -1)
            img_batch = resize_img(img_batch, shape=shape)
            img_batch = np.expand_dims(img_batch, 0)
        else:
            raise Exception('Incorrect shape of input batch')


        with tf.Session(graph=self.graph) as sess:
            out = sess.run('graph/output:0', feed_dict={'graph/input/Placeholder:0' : img_batch})

        if return_input:
            return out, img_batch

        return out
