# coding: utf-8
import cPickle
import numpy as np
import tensorflow as tf
import tflearn
import os
import time
import sys


def unpickle(file):
    fo = open(file, 'rb')
    dict_ = cPickle.load(fo)
    fo.close()
    return dict_


batch_size = 128
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
    
    # Op to initialize variables
    init_op = tf.global_variables_initializer()
# ### Read data
# * Use first 4 data files as training data and last one as validation

train_x = []
train_y = []
for i in xrange(1, 5):
    dict_ = unpickle('../../cifar-10/cifar-10-batches-py/data_batch_' + str(i))
    if i == 1:
        train_x = np.array(dict_['data'])/255.0
        train_y = dict_['labels']
    else:
        train_x = np.concatenate((train_x, np.array(dict_['data'])/255.0), axis=0)
        train_y.extend(dict_['labels'])

train_y = np.array(train_y)
dict_ = unpickle('../../cifar-10/cifar-10-batches-py/data_batch_5')
valid_x = np.array(dict_['data'])/255.0
valid_y = np.eye(10)[dict_['labels']]
del dict_

epochs = 10 * int(round(40000/batch_size))
losses = []
selected_batches = []

with tf.Session(graph=graph) as session:
    session.run(init_op)
    saver = tf.train.Saver()
    saver.restore(session,'./initial-model')
    sequence = np.load('sequence(batch_size-128).npy')  # The sequence to form batches
    approx_batch = np.load('approx_batch.npy')  # batch used to approximate training set
    train_y = np.eye(10)[train_y]

    i = 1
    # 79 for master and 78 for the rest except last one - 77.5
    cursor_start = batch_size * 79 + (78 * int(sys.argv[1]))
    cursor = cursor_start
    loss_drop = []  # Store drop in loss for approx_batch for each batch

    # Get loss before training on the batch
    tflearn.is_training(False, session=session)
    cr1 = session.run([loss], feed_dict={x: train_x[approx_batch], y: train_y[approx_batch]})
    tflearn.is_training(True, session=session)

    while i <= epochs:

        random_train_x = train_x[sequence]
        random_train_y = train_y[sequence]

        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(train_x))]
        feed_dict = {x: batch_xs, y: batch_ys}
        # Train it on the batch
        _ = session.run([optimizer], feed_dict=feed_dict)
        # Get loss on approx_batch after training
        tflearn.is_training(False, session=session)
        cr2 = session.run([loss], feed_dict={x: train_x[approx_batch], y: train_y[approx_batch]})
        loss_drop.append(cr1[0]-cr2[0])
        if i == 1:
            saver.restore(session,'./initial-model')
        else:
            while not os.path.exists('./prev-model'+str(i % 2)):
                time.sleep(1)
            saver.restore(session, './prev-model'+str(i % 2))

        # 79 for master and 78 for the rest except last one - 77.5
        cursor = (cursor + batch_size) % (cursor_start + batch_size * 78)
        if cursor == 0:
            np.save("loss-drop-"+str(sys.argv[1]), loss_drop)
            cursor = cursor_start
            i += 1
            loss_drop = []  # Reset drop
            while not os.path.exists('./prev-model'+str(i % 2)):
                time.sleep(1)
            saver.restore(session, './prev-model'+str(i % 2))

            # Get loss before training on the batch
            tflearn.is_training(False, session=session)
            cr1 = session.run([loss], feed_dict={x: train_x[approx_batch], y: train_y[approx_batch]})
            tflearn.is_training(True, session=session)
