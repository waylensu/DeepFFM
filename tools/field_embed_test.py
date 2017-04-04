#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

import tensorflow as tf
import os.path as osp
import _init_paths
from field_embed.field_embed_op import field_embed as field_embed_op
import numpy as np
from sklearn.datasets import load_digits

sess = tf.InteractiveSession()

embed_size = 1
field_size = 64
features_size = 64
class_size = 10
limits = list(range(field_size + 1))

weights = tf.Variable(tf.zeros([features_size,field_size,embed_size]))
biases = tf.Variable(tf.zeros([field_size,field_size,embed_size]))

digits = load_digits()
vals = digits.data
label = digits.target
inds = [[0]*field_size for x in range(len(digits.target))]

init = tf.global_variables_initializer()
output=field_embed_op(inds, vals, weights, biases, limits, name=None)

sess.run(init)
result=sess.run(output)
print(result)
print(result[0][0][29][0])
