#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

import tensorflow as tf
import os.path as osp
from field_embed_op import field_embed
import numpy as np

sess = tf.InteractiveSession()

features = [[0,1]]
vals = [[2,2]]
weights = np.ones((4,2,3))
biases = np.ones((2,2,3))

weights[0][0][0]=2

#weights = tf.Variable(tf.zeros([4,2,3]),name='weights')
#biases = tf.Variable(tf.zeros([2,2,3]),name='biases')
embed_size = 3
limits = [0, 2, 4]

init = tf.global_variables_initializer()
output=field_embed(features, vals, weights, biases, limits, embed_size, name=None)

sess.run(init)
result=sess.run(output)
