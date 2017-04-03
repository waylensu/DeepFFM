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
import argparse
import sys
from sklearn.datasets import load_iris
import tensorflow as tf
from networks.deep_ffm import DeepFFM
from networks.read_data import read_data_sets
import os.path
import time
from deepffm_reader import inputs

FLAGS = None

def train():
    train_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    test_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')

    # Load data
    chosen = np.array(range(39))

    # Load limits
    limits_path = os.path.join(FLAGS.data_dir, 'limits.txt')
    limits = [0]
    with open(limits_path) as inFile:
        cols = inFile.readline().strip().split('\t')
        lens = np.array([1] * 13 + list(map(int, cols)))[chosen]
        for l in lens:
            limits.append(limits[-1] + l)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 100, 0.95, staircase = True)
        optimizer = tf.train.AdamOptimizer(lr)
        batch_size = 1000
        inds, vals, labels = inputs(train_file, batch_size, FLAGS.num_epochs)    
        test_inds, test_vals, test_labels = inputs(test_file, batch_size)    
        deepffm = DeepFFM(limits, 8, l2_reg_lambda = 0.0001, NUM_CLASSES = 2, inds = inds, vals = vals, labels = labels)
        train_op = optimizer.minimize(deepffm.loss, global_step = global_step)
        #momentum = 0.9
        #train_op = tf.train.GradientDescentOptimizer(lr).minimize(deepffm.loss)
        #train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(deepffm.loss)

        merged = tf.summary.merge_all()
        sess=tf.Session()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                if step % 10 == 0:
                    start_time = time.time()
                    _, summary, loss_value, accuracy, auc, lr_value = sess.run([train_op, merged, deepffm.loss, deepffm.accuracy, deepffm.auc, lr])
                    duration = time.time() - start_time
                    print('Step %d: loss = %.5f, accuracy = %.5f, auc = %.5f, lr = %.5f.(%.5f sec)' % (step, loss_value, accuracy, auc, lr_value, duration))
                else:
                    _, summary = sess.run([train_op, merged])

                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)

                # Test data
                if step % 100 == 0:
                    deepffm.inds, deepffm.vals, deepffm.labels = [test_inds, test_vals, test_labels]
                    start_time = time.time()
                    summary, loss_value, accuracy, auc = sess.run([merged, deepffm.loss, deepffm.accuracy, deepffm.auc])
                    duration = time.time() - start_time
                    print('\tTest Step %d: loss = %.5f, accuracy = %.5f, auc = %.5f. (%.5f sec)' % (step, loss_value, accuracy, auc, duration))
                    test_writer.add_summary(summary, step)
                    deepffm.inds, deepffm.vals, deepffm.labels = [inds, vals, labels]

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop()

        train_writer.close()
        test_writer.close()

        coord.join(threads)
        sess.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--data_dir', type=str, default='/home/wing/DataSet/criteo/pre/deepffm/downSample',
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/home/wing/Project/DeepFFM/logs/summaries',
                      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
