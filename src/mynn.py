from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import tensorflow as tf
from networks.factory import get_network
from networks.read_data import read_data_sets
from deep_ffm.config import cfg
import pdb 
import numpy 
from field_embed.field_embed_op import field_embed as field_embed_op
import field_embed.field_embed_op_grad
from product_layer.product_layer_op import product_layer as product_layer_op
import product_layer.product_layer_op_grad

class Pnn(object):
    def __init__(self, limits, embed_size=8, l2_reg_lambda=0.0, NUM_CLASSES=2 ):

        field_size = len(limits) -1
        features_size = int(limits[-1])

        self.inds_ph = tf.placeholder(tf.int32, shape=[None, field_size])
        self.vals_ph = tf.placeholder(tf.float32, shape=[None, field_size])
        self.labels_ph = tf.placeholder(tf.int32, shape=[None, 10])

        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)
        l2_loss = tf.constant(0.01)

        ###############field_embed################

        #self.fe_weights = tf.get_variable('fe_weights', [features_size, field_size, embed_size], None, init_weights)
        #self.fe_biases = tf.get_variable('fe_biases',[field_size, field_size, embed_size], None, init_biases )

        #self.fe_output = field_embed_op(self.inds_ph, self.vals_ph, self.fe_weights, self.fe_biases, limits)
        ###############product_layer################
        #self.pl_output = product_layer_op(self.fe_output)
        #self.relu_output = tf.nn.relu(self.pl_output)
        ###############fc1#########################
        #fe_shape = self.fe_output.shape
        #fe_size = fe_shape[1]*fe_shape[2]*fe_shape[3]
        
        #self.concat_output = tf.reshape(self.fe_output, [-1, int(fe_size)] )
        #print(self.concat_output.shape)
        #dim = self.concat_output.get_shape()[-1]

        self.fc1_weights = tf.get_variable('fc1_weights', [field_size, 30], None, init_weights)
        self.fc1_biases = tf.get_variable('fc1_biases', [30], None, init_biases)

        self.relu_output = tf.nn.relu(tf.matmul(self.vals_ph, self.fc1_weights) + self.fc1_biases)
        ##############fc#######################
        dim = self.relu_output.get_shape()[-1]
        self.fc_weights = tf.get_variable('fc_weights', [dim, NUM_CLASSES], None, init_weights)
        self.fc_biases = tf.get_variable('fc_biases', [NUM_CLASSES], None, init_biases)

        self.logits = tf.matmul(self.relu_output, self.fc_weights) + self.fc_biases

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_ph)
        self.loss =  tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
