from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf
from pnn import Pnn
#from mynn import Pnn
from deep_ffm.config import cfg
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_digits


digits = load_digits()
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
field_size = 64
batch_size = len(digits.target)

limits = list(range(field_size + 1))
inds = [[0]*field_size for x in range(batch_size)]

pnn = Pnn(limits, 1, NUM_CLASSES = 10)

#global_step = tf.Variable(0, trainable=False)
#lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
#momentum = cfg.TRAIN.MOMENTUM
#train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(pnn.loss, global_step=global_step)
train_op = tf.train.GradientDescentOptimizer(1).minimize(pnn.loss)

print('init')
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

max_iters = 500

for i in range(max_iters):
    #batch = mnist.train.next_batch(batch_size)
    feed_dict = {pnn.inds_ph: inds, pnn.vals_ph: digits.data, pnn.labels_ph: digits.target}
    loss_value = sess.run([pnn.loss, train_op, pnn.pl_output, pnn.fe_output, pnn.logits], feed_dict=feed_dict)
    print ('iter: %d / %d, train loss: %.4f'%(i+1, max_iters, loss_value[0] ) )
    #print ('iter: %d / %d'%(i+1, max_iters) )

    #print (np.any(np.isnan(loss_value[2])))
    #print (np.any(np.isnan(loss_value[3])))
    #print (np.any(np.isnan(loss_value[4])))
    #print ('fe',loss_value[3].mean())
    #print (loss_value[3].max())
    #print (loss_value[3].min())
    #print ('pl',loss_value[2].mean())
    #print (loss_value[2].max())
    #print (loss_value[2].min())
    #print ('logits',loss_value[4].mean())
    #print (loss_value[4].max())
    #print (loss_value[4].min())
    #print (loss_value[0])
    if(i % 1 == 0):
        correct_prediction = tf.equal(pnn.labels_ph, tf.argmax(pnn.logits,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval(feed_dict={pnn.inds_ph: inds, pnn.vals_ph: digits.data, pnn.labels_ph: digits.target}))
