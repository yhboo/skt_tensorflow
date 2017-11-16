import numpy as np
import tensorflow as tf
from layers import Dense, Conv2d
from base_model import quantization_model

class mlp(object):
    def __init__(self, name):
        self.name = name

        with tf.variable_scope(self.name):
            self.d1 = Dense(784, 512, name = 'dense1')
            self.d2 = Dense(512, 512, name = 'dense2')
            self.d3 = Dense(512, 10, name = 'dense3')



    def get_logit(self, x):
        h = tf.nn.relu(self.d1(x))
        h = tf.nn.relu(self.d2(h))
        logit = self.d3(h)

        return logit

    def compute_loss_acc(self, logit, y):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit, labels = y)
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))

        return loss, acc


    def save_params(self, sess, path):
        for pp in tf.global_variables():
            n_arr = pp.name.split('/')
            n = "_".join(n_arr) + '.npy'
            v = sess.run(pp)
            np.save(path+n, v)
            print(pp.name, ' is saved at ', path + n)

    def load_params(self, sess, path):
        for pp in tf.global_variables():
            n_arr = pp.name.split('/')
            n = "_".join(n_arr) + '.npy'
            v = np.load(path + n)
            sess.run(tf.assign(pp, v))
            print(path + n, ' is loaded at ', pp.name)

class mlp_fixed(quantization_model):
    def __init__(self, name, n_bits_dict, pre_trained_path, opt_name):
        super(mlp_fixed, self).__init__(n_bits_dict, pre_trained_path, opt_name)
        self.name = name

        with tf.variable_scope(self.name):
            self.d1 = Dense(784, 512, name = 'dense1')
            self.d2 = Dense(512, 512, name = 'dense2')
            self.d3 = Dense(512, 10, name = 'dense3')



    def get_logit(self, x):
        h = tf.nn.relu(self.d1(x))
        h = tf.nn.relu(self.d2(h))
        logit = self.d3(h)

        #for pp in tf.global_variables():
        #    self.params.append(pp)
        return logit

    def compute_loss_acc(self, logit, y):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit, labels = y)
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))

        return loss, acc



class lenet5(object):
    def __init__(self, name):
        self.name = name

        with tf.variable_scope(self.name):
            self.c1 = Conv2d(1, 6, kernel_size = (5,5), strides = (1,1), padding = 'same', use_bias = False, name = 'conv1')
            self.c2 = Conv2d(6, 16, (5,5), (1,1), 'valid', use_bias = False, name = 'conv2')
            self.d1 = Dense(400, 120, use_bias = False, name = 'dense1')
            self.d2 = Dense(120, 84, use_bias = False, name = 'dense2')
            self.d3 = Dense(84, 10, use_bias = False, name = 'dense3')

    def get_logit(self, x, phase):
        #conv1
        h = self.c1(x)
        h = tf.contrib.layers.batch_norm(h, center = True, scale = True, is_training = phase, scope = self.name + '/conv1')
        h = tf.nn.relu(h)
        #max_pooling
        h = tf.layers.max_pooling2d(h, pool_size=(2,2), strides = (2,2))

        #conv2
        h = self.c2(h)
        h = tf.contrib.layers.batch_norm(h, center = True, scale = True, is_training = phase, scope = self.name + '/conv2')
        h = tf.nn.relu(h)
        #max_pooling
        h = tf.layers.max_pooling2d(h, pool_size=(2,2), strides=(2,2))

        #flatten
        h = tf.reshape(h, [-1, 400])

        #dense1
        h = self.d1(h)
        h = tf.contrib.layers.batch_norm(h, center = True, scale = True, is_training = phase, scope = self.name + '/dense1')
        h = tf.nn.relu(h)

        #dense2
        h = self.d2(h)
        h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=phase, scope=self.name + '/dense2')
        h = tf.nn.relu(h)

        #dense3
        logit = self.d3(h)

        return logit


    def compute_loss_acc(self, logit, y):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit, labels = y)
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))

        return loss, acc

    def save_params(self, sess, path):
        for pp in tf.global_variables():
            n_arr = pp.name.split('/')
            n = "_".join(n_arr) + '.npy'
            v = sess.run(pp)
            np.save(path+n, v)
            print(pp.name, ' is saved at ', path + n)


class lenet5_fixed(quantization_model):
    def __init__(self, name, n_bits_dict, pre_trained_path, opt_name):
        super(lenet5_fixed, self).__init__(n_bits_dict, pre_trained_path, opt_name)
        self.name = name

        with tf.variable_scope(self.name):
            self.c1 = Conv2d(1, 6, kernel_size = (5,5), strides = (1,1), padding = 'same', use_bias = False, name = 'conv1')
            self.c2 = Conv2d(6, 16, (5,5), (1,1), 'valid', use_bias = False, name = 'conv2')
            self.d1 = Dense(400, 120, use_bias = False, name = 'dense1')
            self.d2 = Dense(120, 84, use_bias = False, name = 'dense2')
            self.d3 = Dense(84, 10, use_bias = False, name = 'dense3')

    def get_logit(self, x, phase):
        #conv1
        h = self.c1(x)
        h = tf.contrib.layers.batch_norm(h, center = True, scale = True, is_training = phase, scope = self.name + '/conv1')
        h = tf.nn.relu(h)
        #max_pooling
        h = tf.layers.max_pooling2d(h, pool_size=(2,2), strides = (2,2))

        #conv2
        h = self.c2(h)
        h = tf.contrib.layers.batch_norm(h, center = True, scale = True, is_training = phase, scope = self.name + '/conv2')
        h = tf.nn.relu(h)
        #max_pooling
        h = tf.layers.max_pooling2d(h, pool_size=(2,2), strides=(2,2))

        #flatten
        h = tf.reshape(h, [-1, 400])

        #dense1
        h = self.d1(h)
        h = tf.contrib.layers.batch_norm(h, center = True, scale = True, is_training = phase, scope = self.name + '/dense1')
        h = tf.nn.relu(h)

        #dense2
        h = self.d2(h)
        h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=phase, scope=self.name + '/dense2')
        h = tf.nn.relu(h)

        #dense3
        logit = self.d3(h)
        #for pp in tf.global_variables():
        #    self.params.append(pp)
        return logit


    def compute_loss_acc(self, logit, y):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit, labels = y)
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))

        return loss, acc



class lenet5_pre_defined(object):
    def __init__(self, name):
        self.name = name

    def get_logit(self, x, phase):
        C1 = tf.layers.conv2d(x, filters=6, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)
        C1 = tf.contrib.layers.batch_norm(C1, center=True, scale=True, is_training=phase)
        C1 = tf.nn.relu(C1)
        S2 = tf.layers.max_pooling2d(C1, pool_size=(2, 2), strides=(2, 2))
        C3 = tf.layers.conv2d(S2, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', use_bias=False)
        C3 = tf.contrib.layers.batch_norm(C3, center=True, scale=True, is_training=phase)
        C3 = tf.nn.relu(C3)
        S4 = tf.layers.max_pooling2d(C3, pool_size=(2, 2), strides=(2, 2))
        S4 = tf.reshape(S4, [-1, 16 * 5 * 5])

        C5 = tf.layers.dense(S4, 120)
        C5 = tf.contrib.layers.batch_norm(C5, center=True, scale=True, is_training=phase)
        C5 = tf.nn.relu(C5)
        F6 = tf.layers.dense(C5, 84)
        F6 = tf.contrib.layers.batch_norm(F6, center=True, scale=True, is_training=phase)
        F6 = tf.nn.relu(F6)
        logit = tf.layers.dense(F6, 10)

        return logit

    def compute_loss_acc(self, logit, y):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y)
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))

        return loss, acc

    def save_params(self, sess, path):
        for pp in tf.global_variables():
            n_arr = pp.name.split('/')
            n = "_".join(n_arr) + '.npy'
            v = sess.run(pp)
            np.save(path+n, v)
            print(pp.name, ' is saved at ', path + n)