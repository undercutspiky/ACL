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


def conv_highway(x, fan_in, fan_out, stride, filter_size, device):

    with tf.device("/gpu:"+str(device)):
        H = tflearn.conv_2d(x, fan_out, filter_size, stride, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                        bias_init='uniform', regularizer='L2')
    H = tflearn.batch_normalization(H)
    H = tf.nn.relu(H)
    # Transform gate
    with tf.device("/gpu:" + str(device)):
        T = tflearn.conv_2d(x, fan_out, filter_size, stride, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                        bias_init=tf.constant(-1.0, shape=[fan_out]), regularizer='L2')
    T = tflearn.batch_normalization(T)
    T = tf.nn.sigmoid(T)
    # Carry gate
    C = 1.0 - T
    res = H * T

    if fan_in != fan_out:
        x_new = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        x_new = tf.pad(x_new, [[0, 0], [0, 0], [0, 0], [(fan_out-fan_in)//2, (fan_out-fan_in)//2]])

        res += C * x_new
        return res, tf.reduce_sum(T)
    return (res + (C * x)), tf.reduce_sum(T)


batch_size = 128

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 3072])
    y = tf.placeholder(tf.float32, [None, 10])
    transform_sum = tf.Variable(0.0)

    x_image = tf.reshape(x, [-1, 32, 32, 3])

    net = tflearn.conv_2d(x_image, 64, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2')
    net = tflearn.batch_normalization(net)
    net = tf.nn.relu(net)

    for ii in xrange(5):
        net, t_s = conv_highway(net, 64, 64, 1, 3, device=0)
        transform_sum += t_s

    net, t_s = conv_highway(net, 64, 128, 2, 3, device=0)
    transform_sum += t_s

    for ii in xrange(7):
        net, t_s = conv_highway(net, 128, 128, 1, 3, device=1)
        transform_sum += t_s

    net, t_s = conv_highway(net, 128, 256, 2, 3, device=1)
    transform_sum += t_s

    for ii in xrange(11):
        net, t_s = conv_highway(net, 256, 256, 1, 3, device=2)
        transform_sum += t_s

    net, t_s = conv_highway(net, 256, 512, 2, 3, device=2)
    transform_sum += t_s

    for ii in xrange(5):
        net, t_s = conv_highway(net, 512, 512, 1, 3, device=3)
        transform_sum += t_s

    with tf.device("/gpu:3"):
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
    lr = tf.train.exponential_decay(0.1, global_step, 6260, 0.1, True)
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

epochs = 50  # 10 * int(round(40000/batch_size)+1)
losses = []
iterations = [0]*len(train_x)
transforms = []
with tf.Session(graph=graph) as session:
    session.run(init_op)
    saver = tf.train.Saver()
    save_path = saver.save(session,'./initial-model')
    sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches

    train_y = np.eye(10)[train_y]

    i = 1
    cursor = 0

    tflearn.is_training(True, session=session)

    while i <= epochs:

        random_train_x = train_x[sequence]
        random_train_y = train_y[sequence]

        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(train_x))]
        feed_dict = {x: batch_xs, y: batch_ys}

        # Train it on the batch
        _ = session.run([optimizer], feed_dict=feed_dict)

        cursor += batch_size
        if cursor > len(train_x):
            tflearn.is_training(False, session=session)
            l_list = []
            ac_list = []
            print "GETTING LOSSES FOR ALL EXAMPLES"
            for iii in xrange(100):
                batch_xs = train_x[iii * 400: (iii + 1) * 400]
                batch_ys = train_y[iii * 400: (iii + 1) * 400]
                feed_dict = {x: batch_xs, y: batch_ys}
                cr = session.run([cross_entropy], feed_dict=feed_dict)
                cr = cr[0]

                # Update iterations
                for j in xrange(len(cr)):
                    if cr[j] > 0.0223 and iterations[j] == i-1:
                        iterations[j] += 1
                # Append losses, activations for batch
                l_list.extend(cr)
            # Append losses, activations for epoch
            losses.append(l_list)

            # Validation test
            print "TESTING ON VALIDATION SET for epoch = " + str(i)
            cor_pred = []
            for iii in xrange(20):
                a = session.run([correct_], feed_dict={x: valid_x[iii * 500:(iii + 1) * 500],
                                                       y: valid_y[iii * 500:(iii + 1) * 500]})
                cor_pred.append(a)
            print "Accuracy = " + str(np.mean(cor_pred))
            tflearn.is_training(True, session=session)
            i += 1

    tflearn.is_training(False, session=session)
    print "GETTING TRANSFORMATIONS FOR ALL EXAMPLES"
    for iii in xrange(100):
        batch_xs = train_x[iii * 400: (iii + 1) * 400]
        batch_ys = train_y[iii * 400: (iii + 1) * 400]
        feed_dict = {x: batch_xs, y: batch_ys}
        cr = session.run([transform_sum], feed_dict=feed_dict)
        cr = cr[0]

        transforms.extend(cr)
    np.save('transforms', transforms)
    np.save('losses', losses)
    np.save('iterations', iterations)