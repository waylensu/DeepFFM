import tensorflow as tf
from field_embed_op import field_embed 
from field_embed_op import field_embed_grad
from product_layer_op import product_layer
from product_layer_op import product_layer_grad

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def field_embed_layer(features, vals, limits, embed_size ):
    features_size = limit[-1]
    field_size = features.shape.as_list()[1]
    weights = weight_variable([features_size, field_size, embed_size])
    biases = bias_variable([field_size, field_size, embed_size])
    return field_embed(features, vals, weights, biases, limits)

def product_layer_layer(input):
    return product_layer(input)
    


def network(limits, embed_size, features, vals):
    #input=''
    field_size = len(limits) - 1;
    features_ = tf.placeholder(tf.float32, shape=[None, field_size])
    vals_ = tf.placeholder(tf.float32, shape=[None, field_size])
    label_ = tf.placeholder(tf.float32, shape=[None,2])


    sess = tf.InteractiveSession()

    field_embed(features, vals, weights, )
