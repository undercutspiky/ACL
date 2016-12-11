# coding: utf-8
import cPickle
import numpy as np
import tensorflow as tf
import tflearn
from heapq import nlargest, nsmallest


def unpickle(file):
    fo = open(file, 'rb')
    dict_ = cPickle.load(fo)
    fo.close()
    return dict_


batch_size = 64
batch_norm = True

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 3072])
    y = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 32, 32, 3])

    #net = tflearn.dropout(x_image, 0.2)
    net = tflearn.conv_2d(x_image, 96, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    if batch_norm:
        net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 96, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    if batch_norm:
        net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 96, 3, 2, 'same', 'relu', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.conv_2d(net, 192, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    if batch_norm:
        net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 192, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    if batch_norm:
        net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 192, 3, 2, 'same', 'relu', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.conv_2d(net, 192, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    if batch_norm:
        net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 192, 1, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    if batch_norm:
        net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 10, 1, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    if batch_norm:
        net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.global_avg_pool(net)

    # Calculate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net, y)
    loss = tf.reduce_mean(cross_entropy)

    # Find all the correctly classified examples
    correct_ = tf.equal(tf.argmax(y, 1), tf.argmax(net, 1))

    # Optimizer with gradient clipping
    global_step = tf.Variable(0)
    lr = tf.train.exponential_decay(0.05, global_step, 400000, 0.1, True)
    optimizer = tf.train.MomentumOptimizer(lr,0.9)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

# ### Read data
# * Use first 4 data files as training data and last one as validation

train_x = []
train_y = []
for i in xrange(1, 5):
    dict_ = unpickle('../cifar-10-batches-py/data_batch_' + str(i))
    if i == 1:
        train_x = np.array(dict_['data'])/255.0
        train_y = np.eye(10)[dict_['labels']]
    else:
        train_x = np.concatenate((train_x, np.array(dict_['data'])/255.0), axis=0)
        train_y = np.concatenate((train_y, np.eye(10)[dict_['labels']]), axis=0)

dict_ = unpickle('../cifar-10-batches-py/data_batch_5')
valid_x = np.array(dict_['data'])/255.0
valid_y = np.eye(10)[dict_['labels']]
del dict_

epochs = 70
losses = []

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)
    np.save('sequence',sequence)
    random_train_x = train_x[sequence]
    random_train_y = train_y[sequence]
    i = 1
    cursor = 0
    while i <= epochs:
        tflearn.is_training(True, session=session)
        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(train_x))]
        feed_dict = {x: batch_xs, y: batch_ys}
        _ = session.run([optimizer], feed_dict=feed_dict)

        cursor += batch_size
        if cursor >= len(train_x):
            cursor = 0
            tflearn.is_training(False, session=session)
            l_list = []
            print "GETTING LOSSES FOR TRAIN SET"
            for iii in xrange(200):
                batch_xs = train_x[iii * 200: (iii + 1) * 200]
                batch_ys = train_y[iii * 200: (iii + 1) * 200]
                feed_dict = {x: batch_xs, y: batch_ys}
                cr = session.run([cross_entropy], feed_dict=feed_dict)

                if len(cr[0]) != 200:
                    print "Length of returned array is = "+len(cr[0])
                # Append losses for batch
                l_list.extend(cr[0])
            # Append losses, activations for epoch
            losses.append(l_list)  # activations.append(ac_list)

            # Validation test
            print "TESTING ON VALIDATION SET for epoch = " + str(i)
            cor_pred = []
            for iii in xrange(100):
                a = session.run([correct_], feed_dict={x: valid_x[iii * 100:(iii + 1) * 100],
                                                       y: valid_y[iii * 100:(iii + 1) * 100]})
                cor_pred.append(a)
            print "Accuracy = " + str(np.mean(cor_pred))
            i += 1
    losses = np.array(losses)
    print losses.shape
    np.save('losses-all-conv-batch_size-64', losses)