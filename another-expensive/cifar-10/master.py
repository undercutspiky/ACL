# coding: utf-8
import cPickle
import numpy as np
import tensorflow as tf
import tflearn
import os
import time
import math


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

epochs = 10
losses = []
with tf.Session(graph=graph) as session:
    session.run(init_op)
    saver = tf.train.Saver()
    save_path = saver.save(session,'./initial-model')
    sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches
    np.save('sequence(batch_size-128)', sequence)
    train_y = np.eye(10)[train_y]

    i = 1
    cursor = 0
    loss_drop = []  # Store drop in loss for approx_batch for each batch

    random_train_x = train_x[sequence]
    random_train_y = train_y[sequence]

    # Get loss on train_data before training on the selected batch
    tflearn.is_training(False, session=session)
    cr1 = []
    for ii in xrange(int(math.ceil(float(len(train_x)) / batch_size))):  # The number of batches
        batch_xs = random_train_x[ii * batch_size: min(((ii + 1) * batch_size), len(train_x))]
        batch_ys = random_train_y[ii * batch_size: min(((ii + 1) * batch_size), len(train_x))]
        feed_dict = {x: batch_xs, y: batch_ys}
        cr = session.run([loss], feed_dict=feed_dict)
        cr1.append(cr[0])
    cr1 = np.array(cr1)
    print cr1.shape
    tflearn.is_training(True, session=session)

    while i <= epochs:

        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(train_x))]
        feed_dict = {x: batch_xs, y: batch_ys}

        # Train it on the batch
        _ = session.run([optimizer], feed_dict=feed_dict)
        # Get loss on train data after training
        tflearn.is_training(False, session=session)
        cr2 = []
        for ii in xrange(int(math.ceil(float(len(train_x)) / batch_size))):  # The number of batches
            batch_xs = random_train_x[ii * batch_size: min(((ii + 1) * batch_size), len(train_x))]
            batch_ys = random_train_y[ii * batch_size: min(((ii + 1) * batch_size), len(train_x))]
            feed_dict = {x: batch_xs, y: batch_ys}
            cr = session.run([loss], feed_dict=feed_dict)
            cr2.append(cr[0])
        cr2 = np.array(cr2)
        tflearn.is_training(True, session=session)
        loss_drop.append(cr1-cr2)
        if i == 1:
            saver.restore(session,'./initial-model')
        else:
            saver.restore(session, './prev-model'+str(i % 2))

        cursor = (cursor + batch_size) % (batch_size * 79)  # 79 for master and 78 for the rest
        if cursor == 0:
            print "Waiting for loss drops from other processes"
            # Wait till data is available from others
            for jj in xrange(3):
                while not os.path.exists("loss-drop-"+str(jj)+".npy"):
                    time.sleep(1)
                try:
                    loss_drop.extend(np.load("loss-drop-"+str(jj)+".npy"))
                except IOError:
                    time.sleep(1)
                    loss_drop.extend(np.load("loss-drop-" + str(jj) + ".npy"))
                os.remove("loss-drop-"+str(jj)+".npy")
            # Gotta save all the losses
            losses.append(loss_drop)
            print np.array(losses).shape
            # Train on the train data
            for ii in xrange(int(math.ceil(float(len(train_x))/batch_size))):  # The number of batches
                batch_xs = random_train_x[ii*batch_size: min(((ii+1)*batch_size), len(train_x))]
                batch_ys = random_train_y[ii*batch_size: min(((ii+1)*batch_size), len(train_x))]
                feed_dict = {x: batch_xs, y: batch_ys}
                _ = session.run([optimizer], feed_dict=feed_dict)

            print "#Epochs  = "+str(i)
            i += 1
            loss_drop = []  # Reset drop

            saver.save(session, './prev-model'+str(i % 2))
            if os.path.exists('./prev-model'+str((i-1) % 2)+'.data-00000-of-00001'):
                os.remove('./prev-model'+str((i-1) % 2)+'.data-00000-of-00001')  # Delete the previous obsolete model
                os.remove('./prev-model' + str((i - 1) % 2) + '.index')
                os.remove('./prev-model' + str((i - 1) % 2) + '.meta')

            # Get loss before training on the batch
            tflearn.is_training(False, session=session)
            cr1 = []
            for ii in xrange(int(math.ceil(float(len(train_x)) / batch_size))):  # The number of batches
                batch_xs = random_train_x[ii * batch_size: min(((ii + 1) * batch_size), len(train_x))]
                batch_ys = random_train_y[ii * batch_size: min(((ii + 1) * batch_size), len(train_x))]
                feed_dict = {x: batch_xs, y: batch_ys}
                cr = session.run([loss], feed_dict=feed_dict)
                cr1.append(cr[0])
            cr1 = np.array(cr1)
            tflearn.is_training(True, session=session)
    np.save('loss-drops', np.array(losses))
