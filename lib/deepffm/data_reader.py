#!/usr/bin/env python
# coding=utf-8

from __future__ import (division,absolute_import,print_function,unicode_literals)
import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'ind': tf.FixedLenFeature([], tf.string),
            'val': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    ind = tf.decode_raw(features['ind'], tf.int32)
    val = tf.decode_raw(features['val'], tf.float32)
    
    ind.set_shape([39])
    val.set_shape([39])

    ind = tf.cast(ind, tf.int32)
    val = tf.cast(val, tf.float32)
    label = tf.cast(features['label'], tf.int64)

    return ind, val, label
  
def inputs(filename, batch_size, num_epochs = None):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        ind, val, label = read_and_decode(filename_queue)
        inds, vals, labels = tf.train.shuffle_batch([ind, val, label], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size, min_after_dequeue=1000)

        return inds, vals, labels

def laod_field_range(path):
    field_range = [0]
    with open(limits_path) as inFile:
        cols = inFile.readline().strip().split('\t')
        lens = list(map(int, cols))
        for l in lens:
            field_range.append(field_range[-1] + l)
    return field_range
