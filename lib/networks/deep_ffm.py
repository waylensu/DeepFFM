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

class DeepFFM():

    def __init__(self, limits, embed_size=8, linear1_size=200, linear2_size=100, l2_reg_lambda=0.0, NUM_CLASSES=2 ):

        field_size = len(limits) -1
        features_size = int(limits[-1])

        self.inds_ = tf.placeholder(tf.int32, shape=[None, field_size])
        self.vals_ = tf.placeholder(tf.float32, shape=[None, field_size])
        self.labels_ = tf.placeholder(tf.int64, shape=[None])

        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.1)
        init_biases = tf.constant_initializer(0.0)
        self.l2_loss = tf.constant(0.00)

        # Field_embed
        with tf.name_scope("field_embed"):
            weights = tf.Variable(init_weights([features_size, field_size, embed_size]), name="weights")
            biases = tf.Variable(init_biases([field_size, field_size, embed_size]), name='biases')
            self.field_embed = field_embed_op(self.inds_, self.vals_, weights, biases, limits)
            self.l2_loss = tf.nn.l2_loss(weights)
            self.l2_loss = tf.nn.l2_loss(biases)

        # Product_layer
        with tf.name_scope("product_layer"):
            self.product_layer = tf.nn.relu(product_layer_op(self.field_embed))

        # Linear 1
        with tf.name_scope("linear1"):
            weights = tf.Variable(init_weights([int(self.product_layer.get_shape()[-1]), linear1_size]), name="weights")
            biases = tf.Variable(init_biases([linear1_size]), name='biases')
            self.linear1 = tf.nn.relu(tf.matmul(self.product_layer, weights) + biases)
            self.l2_loss = tf.nn.l2_loss(weights)
            self.l2_loss = tf.nn.l2_loss(biases)

        # Linear 2
        with tf.name_scope("linear2"):
            weights = tf.Variable(init_weights([linear1_size, linear2_size]), name="weights")
            biases = tf.Variable(init_biases([linear2_size]), name='biases')
            self.linear2 = tf.matmul(self.linear1, weights) + biases
            self.l2_loss = tf.nn.l2_loss(weights)
            self.l2_loss = tf.nn.l2_loss(biases)

        # Softmax
        with tf.name_scope("Softmax"):
            weights = tf.Variable(init_weights([linear2_size, NUM_CLASSES]), name="weights")
            biases = tf.Variable(init_biases([NUM_CLASSES]), name='biases')
            self.logits = tf.nn.softmax(tf.matmul(self.linear2, weights) + biases, name="scores")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.l2_loss = tf.nn.l2_loss(weights)
            self.l2_loss = tf.nn.l2_loss(biases)

        # Loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_)
            self.loss =  tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.labels_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
