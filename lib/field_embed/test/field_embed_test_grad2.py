import tensorflow as tf
import _init_paths
from tensorflow.examples.tutorials.mnist import input_data
from field_embed.field_embed_op import field_embed as field_embed_op
from field_embed import field_embed_op_grad
from field_embed.field_embed_op import field_embed_grad

sess = tf.InteractiveSession()

embed_size = 2
field_size = 1
features_size = 2

inds_ = tf.placeholder(tf.int32, shape=[None, field_size])
vals_ = tf.placeholder(tf.float32, shape=[None, field_size])
y_ = tf.placeholder(tf.int32, shape=[None, field_size * field_size * embed_size])


#init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
#init_biases = tf.constant_initializer(0.0)

#W = tf.get_variable('fe_weights', [features_size, field_size, embed_size], None, init_weights)
#b = tf.get_variable('fe_biases',[field_size, field_size, embed_size], None, init_biases )

W = tf.Variable(tf.zeros([features_size,field_size,embed_size]))
b = tf.Variable(tf.zeros([field_size, field_size, embed_size]))

sess.run(tf.global_variables_initializer())

#y = tf.matmul(x,W) + b
#limits = list(range(field_size + 1))
limits = [0,2]

fe_output = field_embed_op(inds_, vals_, W, b, limits)
fe_shape = fe_output.shape
fe_size = fe_shape[1]*fe_shape[2]*fe_shape[3]
y = tf.reshape(fe_output, [-1, int(fe_size)] )

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)

inds = [[0,1],[0,1],[0,1],[0,1]]
vals = [[1,1],[0,1],[1,0],[0,0]]
#vals = [[0,1],[0,1],[1,0],[1,0]]
#label = [[0,1,0,0],[0,1,0,0],[1,0,0,0],[1,0,0,0]]
label = [[0,0,0,1],[0,0,0,1],[0,1,0,0],[0,1,0,0]]
#data = [[1,1],[1,0],[0,1],[0,0]]
#label = [[0,1],[1,0],[1,0],[1,0]]


#############
inds = [[0]]
vals = [[2]]
#label = [[0,1],[1,0]]

grad = [[[[4,0]]]]
output = field_embed_grad(inds, vals, W, b, grad,limits)
result = sess.run(output)
print(result)
exit()
###########
for i in range(1000):
    if(i % 100 == 0):
        print('w',W.eval())
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print(correct_prediction.eval(feed_dict={inds_: inds, vals_: vals, y_: label}))
        a,a_,b,b_=sess.run([y,y_,tf.argmax(y,1),tf.argmax(y_,1)],feed_dict={inds_: inds, vals_: vals, y_: label})
        #print('a',a,'a_',a_)
        print('a',a)
        print('b',b)
        print(accuracy.eval(feed_dict={inds_: inds, vals_: vals, y_: label}))
    train_step.run(feed_dict={inds_: inds, vals_: vals, y_: label})


#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

######################
