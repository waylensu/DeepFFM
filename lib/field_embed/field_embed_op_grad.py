#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)

import tensorflow as tf
from tensorflow.python.framework import ops
import field_embed_op
import pdb

@ops.RegisterGradient("FieldEmbed")
def _field_embed_grad(op, grads):
  """The gradients for `field_embed`.
  Args:
    op: The `field_embed` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `field_embed` op.
  Returns:
    Gradients with respect to the input of `field_embed`.
  """
  features = op.inputs[0]
  vals = op.inputs[1]
  weights = op.inputs[2]
  biases = op.inputs[3]
  
  limits = op.get_attr('limits')

  # compute gradient
  vals_grad, weights_grad, biases_grad = field_embed_op.field_embed_grad(features, vals, weights, biases, grad, limits)

  return [None, vals_grad, weights_grad, biases_grad]  # List of one Tensor, since we have one input
