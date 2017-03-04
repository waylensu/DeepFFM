#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

import tensorflow as tf
import os.path as osp
from product_layer_op import product_layer
from product_layer_op import product_layer_grad
import numpy as np

sess = tf.InteractiveSession()

bottom_data = np.ones((1,2,2,2))

#bottom_data[0][1][0][0]=2
#init = tf.global_variables_initializer()
#top_data=product_layer(bottom_data)
#sess.run(init)
#result=sess.run(top_data)


top_grad = np.ones((1,3))
#top_grad[0][1]=2
bottom_grad = product_layer_grad(bottom_data,top_grad)
result=sess.run(bottom_grad)

