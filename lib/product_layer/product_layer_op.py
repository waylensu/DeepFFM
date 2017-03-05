#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'product_layer.so')
_product_layer_module = tf.load_op_library(filename)
product_layer = _product_layer_module.product_layer
product_layer_grad = _product_layer_module.product_layer_grad
