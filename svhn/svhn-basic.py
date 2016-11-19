# coding: utf-8
import cPickle
import numpy as np
import tensorflow as tf
import tflearn
from heapq import nlargest, nsmallest


def unpickle(file_name):
    fo = open(file_name, 'rb')
    dict_ = cPickle.load(fo)
    fo.close()
    return dict_


batch_size = 128
batch_norm = True

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])

    net = tflearn.dropout(x, 0.2)
    net = tflearn.conv_2d(net, 96, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
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
    lr = tf.train.exponential_decay(0.05, global_step, 656000, 0.1, True)
    optimizer = tf.train.MomentumOptimizer(lr,0.9)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

# Read Data
train_x = unpickle('train_x_1')
train_x = np.concatenate((train_x, unpickle('train_x_2')), axis=0)
train_y = np.eye(10)[unpickle('train_y')]
valid_x = unpickle('valid_x')
valid_y = np.eye(10)[unpickle('valid_y')]

epochs = 150
losses = []
activations = []
iterations = [0] * 65600

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    i = 1
    cursor = 0
    while i <= epochs:
        tflearn.is_training(True, session=session)
        batch_xs = train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = train_y[cursor: min((cursor + batch_size), len(train_x))]
        feed_dict = {x: batch_xs, y: batch_ys}
        _ = session.run([optimizer], feed_dict=feed_dict)

        cursor = (cursor + batch_size) % 65600
        if cursor == 0:
            tflearn.is_training(False, session=session)
            l_list = []
            ac_list = []
            print "GETTING ACTIVATIONS, ITERATIONS AND LOSSES FOR ALL EXAMPLES"
            for iii in xrange(656):
                batch_xs = train_x[iii * 100: (iii + 1) * 100]
                batch_ys = train_y[iii * 100: (iii + 1) * 100]
                feed_dict = {x: batch_xs, y: batch_ys}
                cr, co = session.run([cross_entropy, correct_], feed_dict=feed_dict)

                # Update iterations
                for j in xrange(len(co)):
                    if not co[j]:
                        iterations[cursor + j] += 1
                # Append losses, activations for batch
                l_list.extend(cr)
                ac_list.extend(co)
            # Append losses, activations for epoch
            losses.append(l_list)  # ;activations.append(ac_list)

            # Validation test
            print "TESTING ON VALIDATION SET for epoch = " + str(i)
            cor_pred = []
            for iii in xrange(100):
                a = session.run([correct_], feed_dict={x: valid_x[iii * 100:min(len(valid_x), (iii + 1) * 100)],
                                                       y: valid_y[iii * 100:min(len(valid_x), (iii + 1) * 100)]})
                cor_pred.append(a)
            print "Accuracy = " + str(np.mean(cor_pred))
            i += 1
    losses = np.array(losses)
    iterations = np.array(iterations)  # ;activations = np.array(activations)
    print losses.shape, iterations.shape
    np.save('iterations-all-conv-2', iterations)
    np.save('losses-all-conv-2', losses)