#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

from tensorflow.contrib.learn.python.learn.datasets import base
import os.path as osp
import numpy

class DataSet(object):

    def __init__(self, inds, vals, labels):
        self._num_examples = inds.shape[0]
        self._inds = inds
        self._vals = vals
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size = 1000):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._inds = self._inds[perm]
            self._vals = self._vals[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._inds[start:end], self._vals[start:end], self._labels[start:end]

def read_data_sets(train_dir,
                   validation_size=5000):

    inds = numpy.loadtxt(osp.join(train_dir, 'train_ind.txt'))
    vals = numpy.loadtxt(osp.join(train_dir, 'train_val.txt'))
    labels_ = numpy.loadtxt(osp.join(train_dir, 'train_label.txt'))
    labels = numpy.array([ [0,1] if x == 1. else [1,0] for x in labels_])

    validation_inds = inds[:validation_size]
    validation_vals = vals[:validation_size]
    validation_labels = labels[:validation_size]
    train_inds = inds[validation_size:]
    train_vals = vals[validation_size:]
    train_labels = labels[validation_size:]

    train = DataSet(train_inds, train_vals, train_labels)
    validation = DataSet(validation_inds, validation_vals, validation_labels)

    return base.Datasets(train=train, validation=validation, test=None)
