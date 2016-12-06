# coding: utf-8
import cPickle
import numpy as np
import tensorflow as tf
import tflearn
import os
import time


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
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    saver.save(session,'initial-model')
    sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches

    approx_batch = []  # batch used to approximate training set
    # Create this approx batch by taking 20 examples from each class
    for ii in xrange(10):
        ll = len(np.where(train_y == ii)[0])
        seq = np.random.randint(ll, size=int(round(0.0077*ll)))
        approx_batch.extend(np.where(train_y == ii)[0][seq])
    # Insert one element each of approx_batch in first 310 batches so that no batch has undue advantage
    # 310 cuz for batch size of 128 there are 313 batches so sorry last 3 batches, mi dispiache !
    for i in xrange(200):
        index = i*128
        ran = np.random.randint(128, size=1)
        b = np.where(sequence == approx_batch[i])[0][0]
        sequence[b], sequence[index + ran] = sequence[index + ran], sequence[b]
    np.save('approx_batch',approx_batch)
    np.save('sequence(batch_size-128)', sequence)
    train_y = np.eye(10)[train_y]

    i = 1
    cursor = 0
    loss_drop = []  # Store drop in loss for approx_batch for each batch
    while i <= epochs:

        random_train_x = train_x[sequence]
        random_train_y = train_y[sequence]

        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(train_x))]
        feed_dict = {x: batch_xs, y: batch_ys}
        # Get loss before training on the batch
        tflearn.is_training(False, session=session)
        cr1 = session.run([cross_entropy], feed_dict={x: train_x[approx_batch], y: train_y[approx_batch]})
        tflearn.is_training(True, session=session)
        # Train it on the batch
        _ = session.run([optimizer], feed_dict=feed_dict)
        # Get loss on approx_batch after training
        tflearn.is_training(False, session=session)
        cr2 = session.run([cross_entropy], feed_dict={x: train_x[approx_batch], y: train_y[approx_batch]})
        loss_drop.append(cr1[0][0]-cr2[0][0])
        if i == 1:
            saver.restore(session,'initial-model')
        else:
            saver.restore(session, 'prev-model'+str(i % 2))

        cursor = (cursor + batch_size) % (batch_size * 53)  # 53 for master and 52 for the rest except last one - 51.5
        if cursor == 0:
            print "Waiting for loss drops from other processes"
            # Wait till data is available from others
            for jj in xrange(5):
                while not os.path.exists("loss-drop-"+str(jj)+".npy"):
                    time.sleep(1)
                try:
                    loss_drop.extend(np.load("loss-drop-"+str(jj)+".npy"))
                except IOError:
                    time.sleep(1)
                    loss_drop.extend(np.load("loss-drop-" + str(jj) + ".npy"))
                os.remove("loss-drop-"+str(jj)+".npy")
            # Find the index of the batch with highest drop on approx_batch and train on it
            print len(loss_drop)
            loss_drop = sorted(zip(loss_drop, range(len(loss_drop))), reverse=True)
            selected_batches.append(loss_drop[0][1])
            np.save('selected_batches', selected_batches)
            sel_st_idx = batch_size * loss_drop[0][1]
            batch_xs = random_train_x[sel_st_idx: min((sel_st_idx + batch_size), len(train_x))]
            batch_ys = random_train_y[sel_st_idx: min((sel_st_idx + batch_size), len(train_x))]
            feed_dict = {x: batch_xs, y: batch_ys}
            _ = session.run([optimizer], feed_dict=feed_dict)

            print "#Batches model has been trained on  = "+str(i)
            i += 1
            loss_drop = []  # Reset drop
            saver.save(session, 'prev-model'+str(i % 2))
            if os.path.exists('prev-model'+str((i-1) % 2)):
                os.remove('prev-model'+str((i-1) % 2))  # Delete the previous obsolete model
    np.save('selected_batches',selected_batches)
