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

'''
digits = load_digits()
field_size = 64
batch_size = len(digits.target)
limits = list(range(field_size + 1))
inds = [[0]*field_size for x in range(batch_size)]
'''

# Load data
cate_chosen = [ x+ 13 for x in [0,1,4,5,7,8,13,16,19,21,22,24] ]
int_chosen = list(range(13))
chosen = np.array(int_chosen+cate_chosen)
data_set = read_data_sets('/Users/wing/DataSet/criteo/sample',chosen)
#data_set = read_data_sets('/Volumes/Untitled/data_set/pre',chosen)

# Load limits
limits_path = '/Volumes/Untitled/data_set/pre/limits.txt'
limits = [0]
with open(limits_path) as inFile:
    cols = inFile.readline().strip().split('\t')
    lens = np.array([1]*13+list(map(int,cols)))[chosen]
    for l in lens:
        limits.append(limits[-1] + l)

deepffm = DeepFFM(limits, 1, NUM_CLASSES = 10)

train_op = tf.train.GradientDescentOptimizer(1).minimize(deepffm.loss)

print('init')
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

max_iters = 100
batch_size = 1000

for i in range(max_iters):
    inds, vals, labels = data_set.train.next_batch(batch_size)
    feed_dict = {deepffm.inds_: inds, deepffm.vals_: vals, deepffm.labels_: labels}
    result = sess.run([train_op, deepffm.loss, deepffm.accuracy], feed_dict=feed_dict)
    print ('iter: %d / %d, train loss: %.4f, accuracy: %.4f. '%(i+1, max_iters, result[1], result[2] ) )
    if (i+1) % (10) == 0:
        inds, vals, labels = data_set.validation.next_batch(batch_size)
        feed_dict = {deepffm.inds_: inds, deepffm.vals_: vals, deepffm.labels_: labels}
        result = sess.run([deepffm.loss, deepffm.accuracy], feed_dict=feed_dict)
        print ('\t\titer: %d / %d, test loss: %.4f, accuracy: %.4f. '%(i+1, max_iters, result[0], result[1] ) )
