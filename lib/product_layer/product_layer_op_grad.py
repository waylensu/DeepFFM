#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

import tensorflow as tf
from tensorflow.python.framework import ops
from product_layer import product_layer_op
import pdb

@ops.RegisterGradient("ProductLayer")
def _product_layer_grad(op, grads):
  """The gradients for `product_layer`.
  Args:
    op: The `product_layer` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `product_layer` op.
  Returns:
    Gradients with respect to the input of `product_layer`.
  """
  bottom_data = op.inputs[0]

  # compute gradient
  bottom_grad = product_layer_op.product_layer_grad(bottom_data, grads)

  return [bottom_grad]  # List of one Tensor, since we have one input
