#!/usr/bin/env python
# coding=utf-8

# =============================================================
#  File Name : deep_ffm.py
#  Author : waylensu
#  Mail : waylensu@163.com
#  Created Time : Tue Mar 28 20:18:15 2017
# =============================================================

from __future__ import (division,absolute_import,print_function,unicode_literals)
import tensorflow as tf
from field_embed.field_embed_op import field_embed as field_embed_op
import field_embed.field_embed_op_grad
from product_layer.product_layer_op import product_layer as product_layer_op
import product_layer.product_layer_op_grad
from tensorflow.contrib.metrics import streaming_auc
from tensorflow.contrib.keras import layers as keras
from easydict import EasyDict as edict

def variable_summaries(var):
"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class DeepFFM():


    def __init__(self, limits, embed_size=8, fc1_size=200, fc2_size=100, l2_reg_lambda=0.0, NUM_CLASSES=2, inds=None, vals=None, labels=None):

        field_size = len(limits) - 1
        features_size = int(limits[-1])

        if inds == None:
            self.inds_ = tf.placeholder(tf.int32, shape=[None, field_size])
        else:
            self.inds_ = inds

        if vals == None:
            self.vals_ = tf.placeholder(tf.float32, shape=[None, field_size])
        else:
            self.vals_ = vals

        if labels == None:
            self.labels_ = tf.placeholder(tf.int64, shape=[None])
        else:
            self.labels_ = labels

        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.1)
        init_biases = tf.constant_initializer(0.1)
        self.l2_loss = tf.constant(0.00)
        epsilon = 1e-3

        '''
        # Linear feature
        with tf.name_scope("linear1"):
            batch_size = int(self.inds.get_shape()[0])
            sp_inds = tf.reshape(tf.add(inds + [limits[:-1]] * batch_size), [-1, field_size, 1])
            sparse_input = tf.SparseTensor(sp_inds, vals, [limits[-1]])
            weights = tf.Variable(init_weights([features_size, fc1_size]), name="weights")
            z_BN = tf.matmul(self.sparse_input, weights)
            batch_mean, batch_var = tf.nn.moments(z_BN, [0])
            scale = tf.Variable(tf.ones([fc1_size]))
            beta = tf.Variable(tf.zeros([fc1_size]))
            prelu = keras.PReLU()
            self.linear1 = prelu(tf.nn.batch_normalization(z_BN, batch_mean, batch_var, beta, scale, epsilon))
            self.l2_loss = tf.nn.l2_loss(weights)
            '''

        # Field_embed
        with tf.name_scope("field_embed"):
            weights = tf.Variable(init_weights([features_size, field_size, embed_size]), name="weights")
            biases = tf.Variable(init_biases([field_size, field_size, embed_size]), name='biases')

            self.field_embed = self.act_summary(field_embed_op(self.inds_, self.vals_, weights, biases, limits))

            self.l2_loss = tf.nn.l2_loss(weights)
            self.l2_loss = tf.nn.l2_loss(biases)
            variable_summaries(weights)
            variable_summaries(biases)

        # Product_layer
        with tf.name_scope("product_layer"):
            self.product_layer = self.act_summary(product_layer_op(self.field_embed))

        # Full Connect 1 with BN
        with tf.name_scope("fc1"):
            product_layer_size = int(self.product_layer.get_shape()[-1])
            weights = tf.Variable(init_weights([product_layer_size, fc1_size]), name="weights")
            z_BN = tf.matmul(self.product_layer, weights)
            batch_mean, batch_var = tf.nn.moments(z_BN, [0])
            scale = tf.Variable(tf.ones([fc1_size]))
            beta = tf.Variable(tf.zeros([fc1_size]))
            self.fc1 = self.act_summary(tf.nn.batch_normalization(z_BN, batch_mean, batch_var, beta, scale, epsilon))

            self.l2_loss = tf.nn.l2_loss(weights)
            variable_summaries(weights)

        # Linear 2 with BN
        with tf.name_scope("fc2"):
            weights = tf.Variable(init_weights([fc1_size, fc2_size]), name="weights")
            z_BN = tf.matmul(self.fc1, weights)
            batch_mean, batch_var = tf.nn.moments(z_BN, [0])
            scale = tf.Variable(tf.ones([fc2_size]))
            beta = tf.Variable(tf.zeros([fc2_size]))
            self.fc2 = self.act_summary(tf.nn.batch_normalization(z_BN, batch_mean, batch_var, beta, scale, epsilon))
            self.l2_loss = tf.nn.l2_loss(weights)
            variable_summaries(weights)

        # Softmax
        with tf.name_scope("Softmax"):
            weights = tf.Variable(init_weights([fc2_size, NUM_CLASSES]), name="weights")
            biases = tf.Variable(init_biases([NUM_CLASSES]), name='biases')
            self.logits = tf.nn.softmax(tf.matmul(self.fc2, weights) + biases, name="scores")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

            self.l2_loss = tf.nn.l2_loss(weights)
            self.l2_loss = tf.nn.l2_loss(biases)
            variable_summaries(weights)
            variable_summaries(biases)
            tf.summary.scalar('cross_entropy', self.logits)

        # Loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_)
            self.loss =  tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.labels_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)

        # Auc
        with tf.name_scope("auc"):
            _,self.auc = streaming_auc(self.logits[:,1], self.labels_)


    def act_summary(self, input_tensor, act=keras.PReLU()):
        tf.summary.histogram('pre_activations', input_tensor)
        prelu = keras.PReLU()
        activations = prelu(preactivate)
        tf.summary.histogram('activations', activations)
        return activations
