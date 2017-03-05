#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'field_embed.so')
_field_embed_module = tf.load_op_library(filename)
field_embed = _field_embed_module.field_embed
field_embed_grad = _field_embed_module.field_embed_grad
