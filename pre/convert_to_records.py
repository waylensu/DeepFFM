#!/usr/bin/env python
# coding=utf-8

# =============================================================
#  File Name : convert_to_records.py
#  Author : waylensu
#  Mail : waylensu@163.com
#  Created Time : 2017年04月02日 星期日 15时59分42秒
# =============================================================

from __future__ import (division,absolute_import,print_function,unicode_literals)

import argparse
import os
import os.path
import sys

import tensorflow as tf
import numpy as np

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():

    path = '/home/wing/DataSet/criteo/pre/deepffm/downSample'
    ind_file = open(os.path.join(path, 'train_ind.txt'))
    val_file = open(os.path.join(path, 'train_val.txt'))
    label_file = open(os.path.join(path, 'train_label.txt'))

    train_name = os.path.join(path, 'train.tfrecords')
    train_writer = tf.python_io.TFRecordWriter(train_name)
    test_name = os.path.join(path, 'test.tfrecords')
    test_writer = tf.python_io.TFRecordWriter(test_name)

    for i, (ind, val, label) in enumerate(zip(ind_file, val_file, label_file)):
        if (i % 100000 == 0):
            print(i)
        ind_raw = np.array(list(map(int, ind.strip().split('\t'))), dtype = 'int32').tostring()
        val_raw = np.array(list(map(float, val.strip().split('\t'))), dtype = 'float32').tostring()
        label_raw = int(label.strip())
        example = tf.train.Example(features=tf.train.Features(feature={
            'ind': _bytes_feature(ind_raw),
            'val': _bytes_feature(val_raw),
            'label': _int64_feature(label_raw)
            }))
        if i < 23487305 / 5:
            test_writer.write(example.SerializeToString())
        else:
            train_writer.write(example.SerializeToString())

    train_writer.close()
    test_writer.close()
    ind_file.close()
    val_file.close()
    label_file.close()

if __name__ == '__main__':
    main()
