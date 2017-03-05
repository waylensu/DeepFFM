from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import tensorflow as tf
from networks.factory import get_network
from networks.read_data import read_data_sets
from deep_ffm.config import cfg
import pdb

network = get_network()
data_set = read_data_sets('/Volumes/Untitled/data_set/pre/sample')
with tf.Session() as sess:

    cls_score = network.get_output('fc4')
    label = network.get_output('label')
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = cls_score))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = cls_score))
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                    cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
    momentum = cfg.TRAIN.MOMENTUM
    train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
    print('init')
    sess.run(tf.global_variables_initializer())
    train = data_set.train
    valid = data_set.validation
    max_iters = 1000

    for iter in range(max_iters):
        # get one batch
        inds, vals, labels = train.next_batch()

        # Make one SGD update
        feed_dict = {network.inds: inds, network.vals: vals, network.labels: labels}
        run_options = None
        run_metadata = None
        #pdb.set_trace()
        loss_value = sess.run([loss, train_op], feed_dict=feed_dict)
        if (iter+1) % (1) == 0:
            print ('iter: %d / %d, train loss: %.4f'%(iter+1, max_iters, loss_value[0] ))
        if (iter+1) % (10) == 0:
            inds, vals, labels = valid.next_batch()
            feed_dict = {network.inds: inds, network.vals: vals, network.labels: labels}
            loss_value = sess.run([loss], feed_dict=feed_dict)
            print ('\t\titer: %d / %d, valid loss: %.4f'%(iter+1, max_iters, loss_value[0] ))
