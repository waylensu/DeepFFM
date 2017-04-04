#!/usr/bin/env python
# coding=utf-8

# =============================================================
#  File Name : startFFM.py
#  Author : waylensu
#  Mail : waylensu@163.com
#  Created Time : Tue Mar 28 21:08:45 2017
# =============================================================

from __future__ import (division,absolute_import,print_function,unicode_literals)
import _init_paths
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf
from networks.deep_ffm import DeepFFM
from networks.read_data import read_data_sets
import os.path
from sklearn.datasets import load_digits

digits = load_digits()
field_size = 64
batch_size = len(digits.target)
limits = list(range(field_size + 1))
inds = [[0]*field_size for x in range(batch_size)]
vals = digits.data
labels = digits.target

deepffm = DeepFFM(limits, 4, NUM_CLASSES = 10)

lr = tf.Variable(0.1)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(deepffm.loss)
#momentum = 0.9
#lr = tf.train.exponential_decay(0.5, tf.Variable(0, trainable=False), 500, 0.1, staircase=True)
#train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(deepffm.loss)

print('init')
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

max_iters = 100
#batch_size = 1000

for i in range(max_iters):
    #inds, vals, labels = data_set.train.next_batch(batch_size)
    feed_dict = {deepffm.inds_: inds, deepffm.vals_: vals, deepffm.labels_: labels}
    result = sess.run([train_op, deepffm.loss, deepffm.accuracy, deepffm.auc], feed_dict = feed_dict)
    print ('iter: %d / %d, train loss: %.4f, accuracy: %.4f, auc: %.4f, lr: %.4f.'%(i + 1, max_iters, result[1], result[2], result[3], lr.eval()))
    if (i + 1) % (5) == 0:
        #inds, vals, labels = data_set.validation.next_batch(batch_size)
        feed_dict = {deepffm.inds_: inds, deepffm.vals_: vals, deepffm.labels_: labels}
        #sess.run(tf.local_variables_initializer())
        result = sess.run([deepffm.loss, deepffm.accuracy, deepffm.auc], feed_dict = feed_dict)
        print ('\t\titer: %d / %d, test loss: %.4f, accuracy: %.4f, auc: %.4f. '%(i+1, max_iters, result[0], result[1], result[2] ) )
        #for _ in range(5):
            #result = sess.run([deepffm.auc], feed_dict = feed_dict)
            #print(result[0])

