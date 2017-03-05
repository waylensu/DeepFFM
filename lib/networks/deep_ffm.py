#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

from networks.network import Network
import tensorflow as tf
import numpy 

class Deepffm(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.load_limits()
        field_size = len(self.limits) -1
        self.inds = tf.placeholder(tf.int32, shape=[None, field_size])
        self.vals = tf.placeholder(tf.float32, shape=[None, field_size])
        self.labels = tf.placeholder(tf.int32, shape=[None, 2])
        self.layers = dict({'inds':self.inds, 'vals':self.vals, 'label': self.labels})
        self.trainable = trainable
        self.setup()

    def load_limits(self,limits_path = '/Volumes/Untitled/data_set/pre/limits.txt'):
        with open(limits_path) as inFile:
            cols = inFile.readline().strip().split('\t')
            cate_chosen = [ x+ 13 for x in [0,1,4,5,7,8,13,16,19,21,22,24] ]
            int_chosen = list(range(13))
            chosen = numpy.array(int_chosen+cate_chosen)
            lens = numpy.array([1]*13+list(map(int,cols)))[chosen]
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
