
# coding: utf-8
import cPickle
import numpy as np
import tensorflow as tf
from heapq import nlargest,nsmallest

def unpickle(file):
    fo = open(file, 'rb')
    dict_ = cPickle.load(fo)
    fo.close()
    return dict_

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


batch_size = 100

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [batch_size, 3072])
    y = tf.placeholder(tf.float32, [batch_size, 10])
    
    # Convolution layer weights
    W_conv1 = weight_variable([3, 3, 3, 64])
    b_conv1 = bias_variable([64])
    W_conv2 = weight_variable([3, 3, 64, 64])
    b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    W_conv4 = weight_variable([3, 3, 64, 64])
    b_conv4 = bias_variable([64])
    
    # Fully connected layers
    W_fc1 = weight_variable([4 * 4 * 64, 256])
    b_fc1 = bias_variable([256])
    W_fc2 = weight_variable([256, 128])
    b_fc2 = bias_variable([128])
    W_fc3 = weight_variable([128, 10])
    b_fc3 = bias_variable([10])
    
    # Reshape image
    x_image = tf.reshape(x, [batch_size,32,32,3])
    
    # Forward pass - CNN
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_pool1 = max_pool_3x3(h_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    h_pool2 = max_pool_3x3(h_conv4)
    
    # Forward pass - Fully Connected layer
    h_pool2_flat = tf.reshape(h_pool2, [batch_size, 4*4*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    logits = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        
    # Calculate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    loss = tf.reduce_mean(cross_entropy)
    
    # Find all the correctly classified examples
    correct_ = tf.equal(tf.argmax(y,1), tf.argmax(logits,1))
    
    # Optimizer with gradient clipping
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    
    # Validation test
    x_v = tf.placeholder(tf.float32, [10000, 3072])
    y_v = tf.placeholder(tf.float32, [10000, 10])
    
    x_v_image = tf.reshape(x_v, [10000,32,32,3])
    # Forward propagation
    # Forward pass - CNN
    h_conv1_v = tf.nn.relu(conv2d(x_v_image, W_conv1) + b_conv1)
    h_conv2_v = tf.nn.relu(conv2d(h_conv1_v, W_conv2) + b_conv2)
    h_pool1_v = max_pool_3x3(h_conv2_v)
    h_conv3_v = tf.nn.relu(conv2d(h_pool1_v, W_conv3) + b_conv3)
    h_conv4_v = tf.nn.relu(conv2d(h_conv3_v, W_conv4) + b_conv4)
    h_pool2_v = max_pool_3x3(h_conv4_v)
    
    # Forward pass - Fully Connected layer
    h_pool2_flat_v = tf.reshape(h_pool2_v, [10000, 4*4*64])
    h_fc1_v = tf.nn.relu(tf.matmul(h_pool2_flat_v, W_fc1) + b_fc1)
    h_fc2_v = tf.nn.relu(tf.matmul(h_fc1_v, W_fc2) + b_fc2)
    logits_v = tf.nn.relu(tf.matmul(h_fc2_v, W_fc3) + b_fc3)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y_v,1), tf.argmax(logits_v,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    


# ### Read data
# * Use first 4 data files as training data and last one as validation 

train_x = []
train_y = []
for i in xrange(1,5):
    dict_ = unpickle('cifar-10-batches-py/data_batch_'+str(i))
    if i == 1:
        train_x = dict_['data']
        train_y = np.eye(10)[dict_['labels']]
    else:
        train_x = np.concatenate((train_x, dict_['data']), axis=0)
        train_y = np.concatenate((train_y, np.eye(10)[dict_['labels']]), axis=0)

dict_ = unpickle('cifar-10-batches-py/data_batch_5')
valid_x = dict_['data']
valid_y = np.eye(10)[dict_['labels']]
del dict_


epochs = 400
losses = []; activations = []; iterations = [0]*40000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    i = 1
    cursor = 0
    while i <= epochs:
        batch_xs = train_x[cursor:(cursor+batch_size)]
        batch_ys = train_y[cursor:(cursor+batch_size)]
        feed_dict = {x: batch_xs, y: batch_ys}
        _, cr, co = session.run([optimizer, cross_entropy, correct_], feed_dict = feed_dict)
        losses.extend(cr)
        
        if i > 1:
            for j in xrange(len(co)):
                if co[j] == False:
                    iterations[cursor+j] += 1
                    
        cursor = (cursor + batch_size) % 40000
        if cursor == 0:
            print "TESTING ON VALIDATION SET for epoch = "+str(i)
            a = session.run([accuracy], feed_dict={x_v: valid_x, y_v: valid_y})
            print "Accuracy = "+str(a)
            i += 1

