# coding: utf-8
import cPickle
import numpy as np
import tensorflow as tf
import tflearn
import sys


def unpickle(file):
    fo = open(file, 'rb')
    dict_ = cPickle.load(fo)
    fo.close()
    return dict_


def relu(x, leakiness=0.1):
    return tf.select(tf.less(x, 0.0), leakiness * x, x)


def gaussian_noise_layer(input_layer, std):
    gaussian = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + gaussian


def conv_highway(x, fan_in, fan_out, stride, filter_size, not_pool=False):

    # First layer
    H = tflearn.batch_normalization(x)
    H = relu(H)
    H = tflearn.conv_2d(H, fan_out, filter_size, stride, 'same', 'linear',
                        weights_init=tflearn.initializations.xavier(),
                        bias_init='uniform', regularizer='L2', weight_decay=0.0002)
    # Second layer
    H = tflearn.batch_normalization(H)
    H = relu(H)
    H = tflearn.conv_2d(H, fan_out, filter_size, 1, 'same', 'linear',
                        weights_init=tflearn.initializations.xavier(),
                        bias_init='uniform', regularizer='L2', weight_decay=0.0002)
    # Transform gate
    T = tflearn.conv_2d(H, fan_out, filter_size, 1, 'same', 'linear',
                        weights_init=tflearn.initializations.xavier(),
                        bias_init=tf.constant(-2.0, shape=[fan_out]),
                        regularizer='L2', weight_decay=0.0002)
    T = tf.nn.sigmoid(T)

    # Carry gate
    C = 1.0 - T
    res = H * T

    if fan_in != fan_out:
        if not not_pool:
            x_new = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
            x_new = tf.pad(x_new, [[0, 0], [0, 0], [0, 0], [(fan_out-fan_in)//2, (fan_out-fan_in)//2]])
        else:
            x_new = tf.pad(x, [[0, 0], [0, 0], [0, 0], [(fan_out - fan_in) // 2, (fan_out - fan_in) // 2]])

        res += C * x_new
        return res, tf.reduce_sum(T, axis=[1,2,3])
    return (res + (C * x)), tf.reduce_sum(T, axis=[1,2,3])

batch_size = 128
width = 1

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])
    transform_sum = tf.Variable(0.0)
    noise = tf.placeholder(tf.float32)

    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_zca_whitening()

    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([32, 32], padding=4)

    net = tflearn.input_data(shape=[None, 32, 32, 3], placeholder=x, data_preprocessing=img_prep,
                             data_augmentation=img_aug)

    net = gaussian_noise_layer(net, noise)

    net = tflearn.conv_2d(net, 16, 3, 1, 'same', 'linear', weights_init=tflearn.initializations.xavier(),
                          bias_init='uniform', regularizer='L2', weight_decay=0.0002)

    net, t_s = conv_highway(net, 16, 16 * width, 1, 3, width > 1)
    transform_sum += t_s

    for ii in xrange(3):
        net, t_s = conv_highway(net, 16 * width, 16 * width, 1, 3)
        transform_sum += t_s

    net, t_s = conv_highway(net, 16 * width, 32 * width, 2, 3)
    transform_sum += t_s

    for ii in xrange(3):
        net, t_s = conv_highway(net, 32 * width, 32 * width, 1, 3)
        transform_sum += t_s

    net, t_s = conv_highway(net, 32 * width, 64 * width, 2, 3)
    transform_sum += t_s

    for ii in xrange(3):
        net, t_s = conv_highway(net, 64 * width, 64 * width, 1, 3)
        transform_sum += t_s

    net = tflearn.batch_normalization(net)
    net = relu(net)
    net = tf.reduce_mean(net, [1, 2])
    net = tflearn.fully_connected(net, 10, activation='linear', weights_init=tflearn.initializations.xavier(),
                                  bias_init='uniform', regularizer='L2', weight_decay=0.0002)

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
    gradients, _ = tf.clip_by_global_norm(gradients, 1)
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
        train_x = np.array(dict_['data'])/255.0
        train_y = dict_['labels']
    else:
        train_x = np.concatenate((train_x, np.array(dict_['data'])/255.0), axis=0)
        train_y.extend(dict_['labels'])

train_y = np.array(train_y)
dict_ = unpickle('../../../cifar-10/cifar-10-batches-py/test_batch')
valid_x = np.array(dict_['data'])/255.0
valid_y = np.eye(10)[dict_['labels']]
train_y = np.eye(10)[train_y]
del dict_

train_x = np.dstack((train_x[:, :1024], train_x[:, 1024:2048], train_x[:, 2048:]))
train_x = np.reshape(train_x, [-1, 32, 32, 3])
valid_x = np.dstack((valid_x[:, :1024], valid_x[:, 1024:2048], valid_x[:, 2048:]))
valid_x = np.reshape(valid_x, [-1, 32, 32, 3])

epochs = 150  # 10 * int(round(40000/batch_size)+1)
learn_rate = 0.1
with tf.Session(graph=graph) as session:
    session.run(init_op)
    sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches
    random_train_x = train_x[sequence]
    random_train_y = train_y[sequence]

    i = 1
    cursor = 0

    while i <= epochs:
        if i == 80:
            learn_rate = 0.01
        elif i == 120:
            learn_rate = 0.001

        batch_xs = random_train_x[cursor: min((cursor + batch_size), len(train_x))]
        batch_ys = random_train_y[cursor: min((cursor + batch_size), len(train_x))]
        if i <= 5:
            feed_dict = {x: batch_xs, y: batch_ys, lr: learn_rate, noise: 0.0}
        else:
            feed_dict = {x: batch_xs, y: batch_ys, lr: learn_rate, noise: np.random.uniform(0, 0.5)}

        # Train it on the batch
        tflearn.is_training(True, session=session)
        _ = session.run([optimizer], feed_dict=feed_dict)

        cursor += batch_size
        if cursor > len(train_x):
            cursor = 0
            tflearn.is_training(False, session=session)
            # Validation test
            print "TESTING ON TEST SET for epoch = " + str(i)
            cor_pred = []
            for iii in xrange(100):
                a = session.run([correct_], feed_dict={x: valid_x[iii * 100:(iii + 1) * 100],
                                                       y: valid_y[iii * 100:(iii + 1) * 100]})
                cor_pred.append(a)
            print "Accuracy = " + str(np.mean(cor_pred))
            tflearn.is_training(True, session=session)
            sequence = np.random.choice(len(train_x), size=len(train_x), replace=False)  # The sequence to form batches
            random_train_x = train_x[sequence]
            random_train_y = train_y[sequence]
            i += 1

    tflearn.is_training(False, session=session)
    for level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        transforms, losses = [], []
        print "GETTING TRANSFORMATIONS FOR ALL NOISE = 0"
        for iii in xrange(500):
            batch_xs = train_x[iii * 100: (iii + 1) * 100]
            batch_ys = train_y[iii * 100: (iii + 1) * 100]
            feed_dict = {x: batch_xs, y: batch_ys, noise: level}
            ts, cr = session.run([transform_sum, cross_entropy], feed_dict=feed_dict)

            transforms.extend(ts)
            losses.extend(cr)
        np.save('transforms-'+str(level), transforms)
        np.save('losses-' + str(level), losses)
