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
import time
from deepffm_reader import inputs

'''
digits = load_digits()
field_size = 64
batch_size = len(digits.target)
limits = list(range(field_size + 1))
inds = [[0]*field_size for x in range(batch_size)]
'''

#path = '/home/wing/DataSet/criteo/pre/deepffm/downSample'
path = '/home/wing/DataSet/criteo/pre/deepffm/downSample/sample'
pre_path = '/home/wing/DataSet/criteo/pre/deepffm/downSample'
train_file = os.path.join(pre_path, 'train.tfrecords')
test_file = os.path.join(pre_path, 'test.tfrecords')

# Load data
#cate_chosen = [x + 13 for x in [0,1,4,5,7,8,13,16,19,21,22,24]]
#cate_chosen = [ x+ 13 for x in [0,1,4,5,6,7,8,9,10,12,13,14,16,17,18,19,21,22,24,25] ]
#int_chosen = list(range(13))
#chosen = np.array(int_chosen + cate_chosen)
chosen = np.array(range(39))
#data_set = read_data_sets(path, chosen)
#data_set = read_data_sets('/home/wing/DataSet/criteo/pre', chosen)

# Load limits
limits_path = os.path.join(path, 'limits.txt')
limits = [0]
with open(limits_path) as inFile:
    cols = inFile.readline().strip().split('\t')
    lens = np.array([1] * 13 + list(map(int, cols)))[chosen]
    for l in lens:
        limits.append(limits[-1] + l)



with tf.Graph().as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(0.01, global_step, 100, 0.95, staircase = True)
    #lr = tf.Variable(0.001)
    optimizer = tf.train.AdamOptimizer(lr)
    #max_iters = 1000
    batch_size = 1000
    num_epochs = 100
    inds, vals, labels = inputs(train_file, batch_size, num_epochs)    
    test_inds, test_vals, test_labels = inputs(test_file, batch_size)    
    deepffm = DeepFFM(limits, 8, l2_reg_lambda = 0.0005, NUM_CLASSES = 2, inds = inds, vals = vals, labels = labels)
    train_op = optimizer.minimize(deepffm.loss, global_step = global_step)
    #momentum = 0.9
    #train_op = tf.train.GradientDescentOptimizer(lr).minimize(deepffm.loss)
    #train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(deepffm.loss)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():

            if step % 10 == 0:
                start_time = time.time()
                _, loss_value, accuracy, auc, lr_value = sess.run([train_op, deepffm.loss, deepffm.accuracy, deepffm.auc, lr])
                duration = time.time() - start_time
                print('Step %d: loss = %.5f, accuracy = %.5f, auc = %.5f, lr = %.5f.(%.5f sec)' % (step, loss_value, accuracy, auc, lr_value, duration))
            else:
                _, = sess.run([train_op])

            # Test data
            if step % 100 == 0:
                deepffm.inds, deepffm.vals, deepffm.labels = [test_inds, test_vals, test_labels]
                start_time = time.time()
                loss_value, accuracy, auc = sess.run([deepffm.loss, deepffm.accuracy, deepffm.auc])
                duration = time.time() - start_time
                print('\tTest Step %d: loss = %.5f, accuracy = %.5f, auc = %.5f. (%.5f sec)' % (step, loss_value, accuracy, auc, duration))
                deepffm.inds, deepffm.vals, deepffm.labels = [inds, vals, labels]

            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (num_epochs, step))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

'''
    for i in range(max_iters):
        #inds, vals, labels = data_set.train.next_batch(batch_size)
        #feed_dict = {deepffm.inds_: inds, deepffm.vals_: vals, deepffm.labels_: labels}
        #result = sess.run([train_op, deepffm.loss, deepffm.accuracy, deepffm.auc], feed_dict = feed_dict)
        result = sess.run([train_op, deepffm.loss, deepffm.accuracy, deepffm.auc])
        print ('iter: %d / %d, train loss: %.4f, accuracy: %.4f, auc: %.4f, lr: %.4f.'%(i + 1, max_iters, result[1], result[2], result[3], lr.eval()))
        #if (i + 1) % (50) == 0:
            #inds, vals, labels = data_set.validation.next_batch(batch_size)
            #feed_dict = {deepffm.inds_: inds, deepffm.vals_: vals, deepffm.labels_: labels}
            #sess.run(tf.local_variables_initializer())
            #result = sess.run([deepffm.loss, deepffm.accuracy, deepffm.auc], feed_dict = feed_dict)
            #print ('\t\titer: %d / %d, test loss: %.4f, accuracy: %.4f, auc: %.4f. '%(i+1, max_iters, result[0], result[1], result[2] ) )
            #for _ in range(5):
            #    result = sess.run([deepffm.auc], feed_dict = feed_dict)
            #    print(result[0])
'''
