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


batch_size = 128

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])

    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_zca_whitening()

    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([32, 32], padding=4)

    net = tflearn.input_data(shape=[None, 32, 32, 3], placeholder=x, data_preprocessing=img_prep,
                             data_augmentation=img_aug)

    net = tflearn.conv_2d(net, 96, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 96, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 96, 3, 2, 'same', 'relu', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.conv_2d(net, 192, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 192, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 192, 3, 2, 'same', 'relu', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.conv_2d(net, 192, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 192, 1, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tflearn.conv_2d(net, 10, 1, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
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
    lr = tf.placeholder(tf.float32)
    optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Op to initialize variables
    init_op = tf.global_variables_initializer()

# ### Read data
# * Use first 4 data files as training data and last one as validation

train_x = []
train_y = []
for i in xrange(1, 6):
    dict_ = unpickle('../../../cifar-10/cifar-10-batches-py/data_batch_' + str(i))
    if i == 1:
        train_x = dict_['data']
        train_y = np.eye(10)[dict_['labels']]
    else:
        train_x = np.concatenate((train_x, dict_['data']), axis=0)
        train_y = np.concatenate((train_y, np.eye(10)[dict_['labels']]), axis=0)

dict_ = unpickle('../../../cifar-10/cifar-10-batches-py/test_batch')
valid_x = dict_['data']
valid_y = np.eye(10)[dict_['labels']]
del dict_

train_x = np.dstack((train_x[:, :1024], train_x[:, 1024:2048], train_x[:, 2048:]))
train_x = np.reshape(train_x, [-1, 32, 32, 3])
valid_x = np.dstack((valid_x[:, :1024], valid_x[:, 1024:2048], valid_x[:, 2048:]))
valid_x = np.reshape(valid_x, [-1, 32, 32, 3])

epochs = 100
losses = []

with tf.Session(graph=graph) as session:
    session.run(init_op)
    sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches
    random_train_x = train_x[sequence]
    random_train_y = train_y[sequence]
    i, cursor, learn_rate = 1, 0, 0.1
    while i <= epochs:
        if i == 80:
            learn_rate = 0.01
        elif i == 120:
            learn_rate = 0.003
        tflearn.is_training(True, session=session)
        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(random_train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(random_train_y))]
        feed_dict = {x: batch_xs, y: batch_ys, lr: learn_rate}
        _ = session.run([optimizer], feed_dict=feed_dict)

        cursor += batch_size
        if cursor > len(random_train_x):
            cursor = 0
            tflearn.is_training(False, session=session)
            l_list = []
            print "GETTING LOSSES FOR ALL EXAMPLES"
            for iii in xrange(500):
                batch_xs = train_x[iii * 100: (iii + 1) * 100]
                batch_ys = train_y[iii * 100: (iii + 1) * 100]
                feed_dict = {x: batch_xs, y: batch_ys}
                cr = session.run([cross_entropy], feed_dict=feed_dict)
                cr = cr[0]

                # Append losses for batch
                l_list.extend(cr)
            # Append losses for epoch
            losses.append(l_list)  # activations.append(ac_list)

            # Validation test
            print "TESTING ON VALIDATION SET for epoch = " + str(i)
            cor_pred = []
            for iii in xrange(100):
                a = session.run([correct_], feed_dict={x: valid_x[iii * 100:(iii + 1) * 100],
                                                       y: valid_y[iii * 100:(iii + 1) * 100]})
                cor_pred.append(a)
            print "Accuracy = " + str(np.mean(cor_pred) * 100)
            if len(losses) >= 2:
                ll = np.array(losses)
                ld = ll[-2,:] - ll[-1,:]
                ld = ld.argsort()
                top_k = ld[:30000]
                random_train_x = train_x[top_k]
                random_train_y = train_y[top_k]
                sequence = np.random.choice(len(random_train_x), size=len(random_train_x),
                                            replace=False)  # The sequence to form batches
                random_train_x = random_train_x[sequence]
                random_train_y = random_train_y[sequence]
            else:
                sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches
                random_train_x = train_x[sequence]
                random_train_y = train_y[sequence]
            i += 1
    losses = np.array(losses)
    print losses.shape
    np.save('losses-all-conv', losses)
