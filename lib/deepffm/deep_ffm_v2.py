#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

from networks.network import Network
import tensorflow as tf
import numpy 

class Deepffm(Network):
    def __init__(self, chosen = None, trainable=True):
        self.inputs = []
        self.load_limits(chosen = chosen)
        field_size = len(self.limits) -1
        self.inds = tf.placeholder(tf.int32, shape=[None, field_size])
        self.vals = tf.placeholder(tf.float32, shape=[None, field_size])
        self.labels = tf.placeholder(tf.int32, shape=[None, 2])
        self.layers = dict({'inds':self.inds, 'vals':self.vals, 'label': self.labels})
        self.trainable = trainable
        self.setup()

    def load_limits(self, limits_path = '/Users/wing/DataSet/criteo/cat/limits.txt', chosen = None):
        with open(limits_path) as inFile:
            cols = inFile.readline().strip().split('\t')
            lens = numpy.array(list(map(int,cols)))
            if isinstance(chosen, numpy.ndarray):
                lens = lens[chosen]
            limits = [0]
            for l in lens:
                limits.append(limits[-1] + l)
        self.limits = limits

    def setup(self):
        (self.feed('inds', 'vals')
                .field_embed(self.limits, name = 'fe1')
                .product_layer(name = 'pl2')
                .fc(1600, name = 'fc3')
                .fc(2, name = 'fc4')
                .softmax(name = 'sf5'))
