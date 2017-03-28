import tensorflow as tf
import _init_paths
from tensorflow.examples.tutorials.mnist import input_data
from field_embed.field_embed_op import field_embed as field_embed_op
from field_embed import field_embed_op_grad
from field_embed.field_embed_op import field_embed_grad
import pdb
from sklearn.datasets import load_digits
import numpy as np

sess = tf.InteractiveSession()

embed_size = 1
field_size = 64
features_size = 64
class_size = 10

inds_ = tf.placeholder(tf.int32, shape=[None, field_size])
vals_ = tf.placeholder(tf.float32, shape=[None, field_size])
y_ = tf.placeholder(tf.int64, shape=[None]) 
init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
init_biases = tf.constant_initializer(0.0)

fe_W = tf.get_variable('fe_weights', [features_size, field_size, embed_size], None, init_weights)
fe_b = tf.get_variable('fe_biases',[field_size, field_size, embed_size], None, init_biases )

#fe_W = tf.Variable(tf.zeros([features_size, field_size, embed_size]))
#fe_b = tf.Variable(tf.zeros([field_size, field_size, embed_size]))

#y = tf.matmul(x,fe_W) + fe_b
limits = list(range(field_size + 1))

fe_output = field_embed_op(inds_, vals_, fe_W, fe_b, limits)
fe_shape = fe_output.shape

fe_size = fe_shape[1]*fe_shape[2]*fe_shape[3]

fc_W = tf.get_variable('fc_W',[fe_size,class_size],None, init_weights)
fc_b = tf.get_variable('fc_b',[class_size],None, init_biases)

reshape_output = tf.reshape(fe_output, [-1, int(fe_size)] )

y = tf.matmul(reshape_output,fc_W) + fc_b

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)

digits = load_digits()
#vals = digits.data.astype('float32').tolist()
vals = digits.data
#vals = vals[:1]
#label = digits.target.astype('float32').tolist()
label = digits.target
#label = label[:1]
#label.dtype = np.int32
inds = [[0]*field_size for x in range(len(digits.target))]
#inds = [[0,1],[0,1],[0,1],[0,1]]
#vals = [[1,1],[0,1],[1,0],[0,0]]
#vals = [[0,1],[0,1],[1,0],[1,0]]
#label = [[0,1,0,0],[0,1,0,0],[1,0,0,0],[1,0,0,0]]
#label = [[1,0,0,0],[0,1,0,0],[1,0,0,0],[0,1,0,0]]
#label = [[1,0],[0,1],[1,0],[0,1]]
#data = [[1,1],[1,0],[0,1],[0,0]]
#label = [[0,1],[1,0],[1,0],[1,0]]

print('init')

sess.run(tf.global_variables_initializer())
###########
for i in range(10):
    #if(i % 1 == 0):
    #print('w',fe_W.eval().mean())
    #print('w',fc_W.eval()[0])
    #result = sess.run([y_,y,fe_output,reshape_output,vals_],feed_dict={inds_: inds, vals_: vals, y_: label})
    result = sess.run([fe_output,vals_,train_step],feed_dict={inds_: inds, vals_: vals, y_: label})

    #print('val',result[1].mean())
    #print('fe_output',result[0].mean())
    #print('fe_output',result[0].sum())
    #print('fe_output',result[0].max())
    #print('fe_output',result[0].min())
    #print('fe_output',result[0].min())
    #print('fe_output',np.any(np.isnan(result[0])))
    #print(result[0])
    #print(result[3].shape,result[4].shape)
    #print(result[3].sum(),result[4].sum())
    correct_prediction = tf.equal(tf.argmax(y,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={inds_: inds, vals_: vals, y_: label}))
    #result = sess.run([train_step],feed_dict={inds_: inds, vals_: vals, y_: label})


#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

######################
