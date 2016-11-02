
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
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


batch_size = 100

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [batch_size, 3072])
    y = tf.placeholder(tf.float32, [batch_size, 10])
    
    # Convolution layer weights
    W_conv1 = tf.Variable(tf.truncated_normal([3,3,3,64], stddev=5e-2))
    b_conv1 = bias_variable([64])
    W_conv2 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=5e-2))
    b_conv2 = bias_variable([64])
    W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=5e-2))
    b_conv3 = bias_variable([64])
    W_conv4 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=5e-2))
    b_conv4 = bias_variable([64])
    W_conv5 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=5e-2))
    b_conv5 = bias_variable([64])
    W_conv6 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=5e-2))
    b_conv6 = bias_variable([64])
    
    # Fully connected layers
    flat_length = 8*8*64
    W_fc1 = tf.Variable(tf.truncated_normal([flat_length, 384], stddev=0.004))
    b_fc1 = bias_variable([384])
    W_fc2 = tf.Variable(tf.truncated_normal([384, 192], stddev=0.004))
    b_fc2 = bias_variable([192])
    W_fc3 = tf.Variable(tf.truncated_normal([192, 10], stddev=1/192.0))
    b_fc3 = bias_variable([10])
    
    # Reshape image
    x_image = tf.reshape(x, [batch_size,32,32,3])
    
    # Forward pass - CNN
    conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #conv2 = tf.nn.relu(conv2d(conv1, W_conv2) + b_conv2)
    #conv3 = tf.nn.relu(conv2d(conv1, W_conv3) + b_conv3)
    pool1 = max_pool_3x3(conv1)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv4 = tf.nn.relu(conv2d(norm1, W_conv4) + b_conv4)
    #conv5 = tf.nn.relu(conv2d(conv4, W_conv5) + b_conv5)
    #conv6 = tf.nn.relu(conv2d(conv4, W_conv6) + b_conv6)
    norm2 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = max_pool_3x3(norm2)
    
    # Forward pass - Fully Connected layer
    pool2_flat = tf.reshape(pool2, [batch_size, flat_length])
    fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)
    fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
    logits = tf.nn.relu(tf.matmul(fc2, W_fc3) + b_fc3)

    # Sum of activations of all hidden units
    total_activation = tf.Variable(tf.zeros([batch_size]))
    total_activation += tf.reduce_sum(fc1, 1)
    total_activation += tf.reduce_sum(fc2, 1)

        
    # Calculate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    loss = tf.reduce_mean(cross_entropy)
    
    # Find all the correctly classified examples
    correct_ = tf.equal(tf.argmax(y,1), tf.argmax(logits,1))
    
    # Optimizer with gradient clipping
    global_step = tf.Variable(0)
    lr = tf.train.exponential_decay(0.1,global_step,40000,0.1,True)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    
    # Validation test
    x_v = tf.placeholder(tf.float32, [100, 3072])
    y_v = tf.placeholder(tf.float32, [100, 10])
    
    x_image_v = tf.reshape(x_v, [100,32,32,3])
    # Forward propagation
    # Forward pass - CNN
    conv1_v = tf.nn.relu(conv2d(x_image_v, W_conv1) + b_conv1)
    #conv2_v = tf.nn.relu(conv2d(conv1_v, W_conv2) + b_conv2)
    #conv3_v = tf.nn.relu(conv2d(conv1_v, W_conv3) + b_conv3)
    pool1_v = max_pool_3x3(conv1_v)
    norm1_v = tf.nn.lrn(pool1_v, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv4_v = tf.nn.relu(conv2d(norm1_v, W_conv4) + b_conv4)
    #conv5_v = tf.nn.relu(conv2d(conv4_v, W_conv5) + b_conv5)
    #conv6_v = tf.nn.relu(conv2d(conv4_v, W_conv6) + b_conv6)
    norm2 = tf.nn.lrn(conv4_v, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2_v = max_pool_3x3(norm2)

    # Forward pass - Fully Connected layer
    pool2_flat_v = tf.reshape(pool2_v, [batch_size, flat_length])
    fc1_v = tf.nn.relu(tf.matmul(pool2_flat_v, W_fc1) + b_fc1)
    fc2_v = tf.nn.relu(tf.matmul(fc1_v, W_fc2) + b_fc2)
    logits_v = tf.nn.relu(tf.matmul(fc2_v, W_fc3) + b_fc3)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y_v,1), tf.argmax(logits_v,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    


# Read data
# Use first 4 data files as training data and last one as validation

train_x = []
train_y = []
for i in xrange(1,5):
    dict_ = unpickle('../cifar-10-batches-py/data_batch_'+str(i))
    if i == 1:
        train_x = dict_['data']
        train_y = np.eye(10)[dict_['labels']]
    else:
        train_x = np.concatenate((train_x, dict_['data']), axis=0)
        train_y = np.concatenate((train_y, np.eye(10)[dict_['labels']]), axis=0)

dict_ = unpickle('../cifar-10-batches-py/data_batch_5')
valid_x = dict_['data']
valid_y = np.eye(10)[dict_['labels']]
del dict_


epochs = 150
losses = []; accuracies=[]

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    i = 1
    cursor = 0
    while i <= epochs:
        batch_xs = train_x[cursor:(cursor+batch_size)]
        batch_ys = train_y[cursor:(cursor+batch_size)]
        feed_dict = {x: batch_xs, y: batch_ys}
        _ = session.run([optimizer], feed_dict = feed_dict)
        
        cursor = (cursor + batch_size) % 40000

        # If epoch is done
        if cursor == 0:
            i += 1
            l_list = [];
            print "GETTING LOSSES FOR TRAIN SET"
            for iii in xrange(len(train_x)/batch_size):
                batch_xs = train_x[iii*batch_size:(iii + 1)*batch_size]
                batch_ys = train_y[iii*batch_size:(iii + 1)*batch_size]
                feed_dict = {x: batch_xs, y: batch_ys}
                cr = session.run([cross_entropy], feed_dict=feed_dict)

                # Append losses, activations for batch
                l_list.extend(cr[0]);
            # Append losses, activations for epoch
            losses.append(l_list)
            print "MEAN LOSS = "+str(np.mean(l_list))
            # Validation test
            print "TESTING ON VALIDATION SET for epoch = " + str(i)
            cor_pred = []
            for iii in xrange(100):
                a = session.run([correct_prediction], feed_dict={x_v: valid_x[iii * 100:(iii + 1) * 100],
                                                                 y_v: valid_y[iii * 100:(iii + 1) * 100]})
                cor_pred.append(a)
            print "Accuracy = " + str(np.mean(cor_pred))
            accuracies.append(np.mean(cor_pred))

            # TRAIN ON HIGH DROP EXAMPLES
            saver.save(session, 'cifar-model')
            if len(losses) > 15: # Check if at least 10 epochs have been done
                hd_loss = []; hd_accuracy = []
                drop = np.array(losses[-2]) - np.array(losses[-1])
                prev_drop = losses[-1]
                while i <= epochs:
                    softmax_prob = np.exp(-drop) / np.sum(np.exp(-drop), axis=0)
                    selected_examples = np.random.choice(len(train_x), len(train_x), replace=False, p=softmax_prob)
                    hd_train_x = train_x[selected_examples]
                    hd_train_y = train_y[selected_examples]
                    for iii in xrange(len(train_x) / batch_size):
                        batch_xs = hd_train_x[iii * batch_size:(iii + 1) * batch_size]
                        batch_ys = hd_train_y[iii * batch_size:(iii + 1) * batch_size]
                        feed_dict = {x: batch_xs, y: batch_ys}
                        _ = session.run([optimizer], feed_dict=feed_dict)
                    # Now get the losses on whole train set
                    print "GETTING LOSSES FOR TRAIN SET"
                    l_list = []
                    for iii in xrange(len(train_x) / batch_size):
                        batch_xs = train_x[iii * batch_size:(iii + 1) * batch_size]
                        batch_ys = train_y[iii * batch_size:(iii + 1) * batch_size]
                        feed_dict = {x: batch_xs, y: batch_ys}
                        cr = session.run([cross_entropy], feed_dict=feed_dict)

                        # Append losses for batch
                        l_list.extend(cr[0])
                    print "TESTING ON VALIDATION SET for epoch = " + str(i)
                    cor_pred = []
                    for iii in xrange(100):
                        a = session.run([correct_prediction], feed_dict={x_v: valid_x[iii * 100:(iii + 1) * 100],
                                                                         y_v: valid_y[iii * 100:(iii + 1) * 100]})
                        cor_pred.append(a)
                    print 'Training set loss = '+str(np.mean(l_list))+' Validation accuracy = '+str(np.mean(cor_pred))
                    losses.append(np.mean(l_list)); accuracies.append(np.mean(cor_pred))
                    drop = np.array(prev_drop) - np.array(l_list)
                    prev_drop = l_list
                    i += 1
    np.save('hd-losses', losses)
    np.save('hd-accuracies', accuracies)